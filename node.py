import logging
import sys

import numpy as np
import torch
import evaluate
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
from utils import init_model, preprocessing_raw_datasets, init_optimizer, init_scheduler
import torch.nn.functional as F


class Client(nn.Module):
    def __init__(self, args, id, model_type, train_dataset=None):
        super().__init__()
        self.args = args
        self.id = id
        self.name = 'client' + str(id)
        self.model_type = model_type

        self.device = args.device
        self.optimizer_type = args.optimizer
        self.scheduler_type = args.scheduler

        self.max_seq_length = args.max_seq_length
        self.batch_size = args.batch_size
        self.temperature = args.temperature

        self.tokenizer, self.model = init_model(self.name, self.model_type, self.args.num_classes)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.train_dataset = train_dataset

        self.E = args.E
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.warmup_steps = args.warmup_steps

    def local_update(self, train_datasets=None):
        self.model.to(self.device).train()
        if train_datasets is not None:
            self.train_dataset = train_datasets

        train_dataset = preprocessing_raw_datasets(self.train_dataset, self.tokenizer, self.max_seq_length)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size,
                                      collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer,
                                   num_warmup_steps=0.06 * len(train_dataloader) * self.args.E,
                                   num_training_steps=len(train_dataloader) * self.E)

        train_loss = 0.
        metric = evaluate.load(self.args.metric_type)
        for batch in tqdm(train_dataloader, desc='Iteration'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            train_loss += loss.item()
            if self.args.num_classes == 1:
                prediction = logits.squeeze()
            else:
                prediction = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=prediction, references=batch['labels'])
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(train_dataloader)
        train_results = metric.compute(average='weighted')
        logging.info('train_loss={}, train_results={}'.format(train_loss, train_results))

        self.model.cpu()

    def local_distillation(self, public_dataset, logits_glob):
        self.model.to(self.device).train()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.tokenizer, self.max_seq_length, logits_glob)
        public_dataloader = DataLoader(public_dataset, shuffle=True, batch_size=self.batch_size,
                                       collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer,
                                   num_warmup_steps=0.06 * len(public_dataloader) * self.args.E,
                                   num_training_steps=len(public_dataloader) * self.E)

        train_loss = 0.
        for batch in tqdm(public_dataloader, desc='Distilling'):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # unlabelled public data
            # del batch['labels']

            soft_label = batch.pop('logits')
            outputs = self.model(**batch)
            logits = outputs.logits

            loss = F.cross_entropy(logits, torch.argmax(soft_label, dim=-1))
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(public_dataloader)
        logging.info('train_loss={}'.format(train_loss))

        self.model.cpu()

    def comm_local_distillation(self, public_dataset, logits_glob):
        self.model.to(self.device).train()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.tokenizer, self.max_seq_length, logits_glob)
        public_dataloader = DataLoader(public_dataset, shuffle=True, batch_size=self.batch_size,
                                       collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer,
                                   num_warmup_steps=0.06 * len(public_dataloader) * self.args.E,
                                   num_training_steps=len(public_dataloader) * self.E)

        train_loss = 0.
        for batch in tqdm(public_dataloader, desc='Distilling'):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # unlabelled public data
            # del batch['labels']

            soft_label = batch.pop('logits')
            outputs = self.model(**batch)
            logits = outputs.logits

            prob_client = F.log_softmax(logits, dim=-1)
            prob_server = F.softmax(soft_label, dim=-1)

            loss = F.kl_div(prob_client, prob_server, reduction='batchmean')

            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(public_dataloader)
        logging.info('train_loss={}'.format(train_loss))

        self.model.cpu()

    def compute_logits(self, public_dataset):
        self.model.to(self.device).eval()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.tokenizer, self.max_seq_length)
        public_dataloader = DataLoader(public_dataset, shuffle=False, batch_size=self.batch_size,
                                       collate_fn=self.data_collator)

        logits = None
        for batch in tqdm(public_dataloader, desc='Predicting'):
            # del batch['labels']
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logit = outputs.logits

            if logits is None:
                logits = logit.detach().cpu()
            else:
                logits = torch.cat([logits, logit.detach().cpu()], dim=0)

        self.model.cpu()
        return logits


class Server(nn.Module):
    def __init__(self, args, id, model_type, public_dataset=None):
        super().__init__()
        self.args = args
        self.id = id
        self.name = 'server'
        self.model_type = model_type

        self.device = args.device
        self.optimizer_type = args.optimizer
        self.scheduler_type = args.scheduler

        self.max_seq_length = args.max_seq_length
        self.batch_size = args.batch_size
        self.temperature = args.temperature

        self.tokenizer, self.model = init_model(self.name, self.model_type, self.args.num_classes)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.public_dataset = public_dataset

    def logit_ensemble_with_norm(self, logits_locals, all_clients_weights, weights=None):

        num_clients = len(all_clients_weights)
        if num_clients == 0:
            raise ValueError("The number of clients is 0 and cannot be evenly distributed.")

        if weights:
            client_scales = {cid: w for cid, w in zip(logits_locals.keys(), weights)}
        else:
            client_scales = {cid: 1.0 / num_clients for cid in logits_locals.keys()}

        logging.info(f'Weight distribution for each client: {client_scales}')

        # Aggregate logits
        logits_shape = next(iter(logits_locals.values())).shape
        ensemble_logits = torch.zeros(logits_shape)

        for cid, logits in logits_locals.items():
            sample_weights = all_clients_weights[cid].view(-1, 1)  # (num_samples, 1)
            # scale = client_scales[cid]
            ensemble_logits += logits * sample_weights * client_scales[cid]

        return ensemble_logits

    def logit_ensemble(self, logits_locals, weights):
        return sum(weight * logit for weight, logit in zip(weights, logits_locals))

    def logit_ensemble_with_ERA(self, logits_locals, weights):
        logits_glob = torch.zeros_like(logits_locals[0])
        for k in range(len(logits_locals)):
            logits_glob += weights[k] * logits_locals[k]
        # logits_glob /= len(logits_locals)
        T = 0.1
        logits_glob = torch.softmax(logits_glob / T, dim=-1)
        return logits_glob

    def centralized_mixed_training(self, centralized_mixed_train_dataset):
        self.model.to(self.device).train()

        centralized_mixed_train_dataset = preprocessing_raw_datasets(centralized_mixed_train_dataset, self.tokenizer,
                                                                     self.max_seq_length)
        train_dataloader = DataLoader(centralized_mixed_train_dataset, shuffle=True, batch_size=self.batch_size,
                                      collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.args.lr, weight_decay=self.args.weight_decay,
                                   momentum=self.args.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer,
                                   num_warmup_steps=0.06 * len(train_dataloader) * self.args.E,
                                   num_training_steps=len(train_dataloader) * self.args.E)

        train_loss = 0.
        metric = evaluate.load(self.args.metric_type)
        for batch in tqdm(train_dataloader, desc='Iteration'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            train_loss += loss.item()
            if self.args.num_classes == 1:
                prediction = logits.squeeze()
            else:
                prediction = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=prediction, references=batch['labels'])

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(train_dataloader)
        train_results = metric.compute(average='weighted')
        logging.info('train_loss={}, train_results={}'.format(train_loss, train_results))

        self.model.cpu()

    def ensemble_distillation(self, public_dataset, logits_glob):
        self.model.to(self.device).train()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.tokenizer, self.max_seq_length, logits_glob)
        public_dataloader = DataLoader(public_dataset, shuffle=True, batch_size=self.batch_size,
                                       collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.args.dis_lr,
                                   weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, num_warmup_steps=0.06 * len(public_dataloader) * self.args.E,
                                   num_training_steps=len(public_dataloader) * self.args.dis_epochs)

        train_loss = 0.
        for batch in tqdm(public_dataloader, desc='Distilling'):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # del batch['labels']

            soft_label = batch.pop('logits')

            outputs = self.model(**batch)
            logits = outputs.logits

            if self.args.algorithm in ['fed_avg', 'fed_avg_modified']:
                loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(soft_label, dim=-1), reduction='batchmean')
            elif self.args.algorithm == 'fed_kd':
                loss = F.mse_loss(logits, soft_label)
            elif self.args.algorithm == 'ds_fl':
                loss = F.cross_entropy(logits, torch.argmax(soft_label, dim=-1))
            else:
                sys.exit("Not implemented algorithm, code exit, re-run to use correct algorithm")
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(public_dataloader)
        logging.info('train_loss={}'.format(train_loss))

        self.model.cpu()

    def mhat_distillation(self, public_dataset, logits_locals, weights):
        self.model.to(self.device).train()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.tokenizer, self.max_seq_length, logits_locals)
        public_dataloader = DataLoader(public_dataset, shuffle=True, batch_size=self.batch_size,
                                       collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.args.dis_lr,
                                   weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, num_warmup_steps=0.06 * len(public_dataloader) * self.args.E,
                                   num_training_steps=len(public_dataloader) * self.args.dis_epochs)

        train_loss = 0.
        for batch in tqdm(public_dataloader, desc='Distilling'):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # unlabelled public data
            # del batch['labels']

            soft_label = []  # local clients predict the logit
            for k in range(len(logits_locals)):
                soft_label.append(batch.pop('logits{}'.format(k)))

            outputs = self.model(**batch)
            logits = outputs.logits

            loss = torch.tensor(0.0, requires_grad=True)
            if self.args.algorithm == 'mhat_ce':
                for k in range(len(soft_label)):
                    temp_ce_loss = weights[k] * F.cross_entropy(logits, torch.argmax(soft_label[k], dim=-1),
                                                                reduction='mean')
                    loss = loss + temp_ce_loss
            elif self.args.algorithm == 'mhat_kl':
                prob_server = F.log_softmax(logits, dim=-1)
                prob_clients = [F.softmax(logits_client, dim=-1) for logits_client in soft_label]
                for k in range(len(soft_label)):
                    temp_kl_loss = weights[k] * F.kl_div(prob_server, prob_clients[k], reduction='batchmean')
                    loss = loss + temp_kl_loss
            else:
                sys.exit("Not implemented algorithm, code exit, re-run to use correct algorithm")

            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(public_dataloader)
        logging.info('train_loss={}'.format(train_loss))

        self.model.cpu()

    def comm_distillation(self, public_dataset, logits_ensemble):
        self.model.to(self.device).train()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.tokenizer, self.max_seq_length,
                                                    logits_ensemble)
        public_dataloader = DataLoader(public_dataset, shuffle=True, batch_size=self.batch_size,
                                       collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.args.dis_lr,
                                   weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer,
                                   num_warmup_steps=0.06 * len(public_dataloader) * self.args.E,
                                   num_training_steps=len(public_dataloader) * self.args.dis_epochs)

        train_loss = 0.
        for batch in tqdm(public_dataloader, desc='Distilling'):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # unlabelled public data
            # del batch['labels']

            soft_label = batch.pop('logits')

            outputs = self.model(**batch)
            logits = outputs.logits

            prob_server = F.log_softmax(logits, dim=-1)
            prob_client = F.softmax(soft_label, dim=-1)

            loss = F.kl_div(prob_server, prob_client, reduction='batchmean')

            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(public_dataloader)
        logging.info('train_loss={}'.format(train_loss))

        self.model.cpu()

    def compute_logits(self, public_dataset):
        self.model.to(self.device).eval()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.tokenizer, self.max_seq_length)
        public_dataloader = DataLoader(public_dataset, shuffle=False, batch_size=self.batch_size,
                                       collate_fn=self.data_collator)

        logits = None
        for batch in tqdm(public_dataloader, desc='Predicting'):
            # del batch['labels']
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logit = outputs.logits

            if logits is None:
                logits = logit.detach().cpu()
            else:
                logits = torch.cat([logits, logit.detach().cpu()], dim=0)

        self.model.cpu()
        return logits
