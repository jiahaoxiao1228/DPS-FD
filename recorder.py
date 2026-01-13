import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import preprocessing_raw_datasets


def validation(node, validation_dataset):
    node.model.to(node.device).eval()

    validation_dataset = preprocessing_raw_datasets(validation_dataset, node.tokenizer, node.max_seq_length)
    validation_dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=node.batch_size, collate_fn=node.data_collator)

    eval_loss = 0.
    metric = evaluate.load(node.args.metric_type)
    for batch in tqdm(validation_dataloader, desc='Evaluating'):
        batch = {k: v.to(node.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = node.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        eval_loss += loss
        if node.args.num_classes == 1:
            prediction = logits.squeeze()
        else:
            prediction = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=prediction, references=batch['labels'])

    eval_loss = eval_loss / len(validation_dataloader)
    eval_results = metric.compute(average='weighted')

    node.model.cpu()
    return eval_loss, eval_results


def test(node, test_dataset, id_test_dataset=None):
    node.model.to(node.device).eval()

    id_test_dataloader = None
    if id_test_dataset is not None:
        id_test_dataset = preprocessing_raw_datasets(id_test_dataset, node.tokenizer, node.max_seq_length)
        id_test_dataloader = DataLoader(id_test_dataset, shuffle=False, batch_size=node.batch_size, collate_fn=node.data_collator)

    test_dataset = preprocessing_raw_datasets(test_dataset, node.tokenizer, node.max_seq_length)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=node.batch_size, collate_fn=node.data_collator)

    id_scores = []
    ood_scores = []
    auroc = None
    fpr95 = None

    # F1 score
    test_loss = 0.
    metric = evaluate.load(node.args.metric_type)
    for batch in tqdm(test_dataloader, desc='OOD Predicting'):
        batch = {k: v.to(node.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = node.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        test_loss += loss
        if node.args.num_classes == 1:
            prediction = logits.squeeze()
        else:
            prediction = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=prediction, references=batch['labels'])

        # Get the maximum value of OOD softmax
        conf = torch.logsumexp(logits, dim=-1)  # energy score
        # softmax = torch.nn.functional.softmax(logits, dim=-1)
        # msp = torch.max(softmax, dim=-1).values
        ood_scores.extend(conf.detach().cpu().numpy())

    # if id_test_dataset is not None:
        # for batch in tqdm(id_test_dataloader, desc='ID Predicting'):
            # batch = {k: v.to(node.device) for k, v in batch.items()}
            # with torch.no_grad():
                # outputs = node.model(**batch)
            # logits = outputs.logits

            # Get the maximum value of ID softmax
            # conf = torch.logsumexp(logits, dim=-1)  # energy score
            # #softmax = torch.nn.functional.softmax(logits, dim=-1)
            # #msp = torch.max(softmax, dim=-1).values
            # id_scores.extend(conf.detach().cpu().numpy())

        # #AUROC 和 FPR95
        # id_scores = np.array(id_scores)
        # ood_scores = np.array(ood_scores)
        # all_scores = np.concatenate([id_scores, ood_scores])
        # labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        # #Calculate statistical measures of ID 
        # id_mean_val = np.mean(id_scores)
        # id_std_val = np.std(id_scores)
        # id_min_val = np.min(id_scores)
        # id_max_val = np.max(id_scores)
        # #Calculate statistical measures of OOD
        # ood_mean_val = np.mean(ood_scores)
        # ood_std_val = np.std(ood_scores)
        # ood_min_val = np.min(ood_scores)
        # ood_max_val = np.max(ood_scores)
        
        # print("Distribution of ID and OOD scores：")
        # print(f"id_Mean: {id_mean_val:.4f}, ood_Mean: {ood_mean_val:.4f}")
        # print(f"id_Std:  {id_std_val:.4f}, ood_Std:  {ood_std_val:.4f}")
        # print(f"id_Min:  {id_min_val:.4f}, ood_Min:  {ood_min_val:.4f}")
        # print(f"id_Max:  {id_max_val:.4f}, ood_Max:  {ood_max_val:.4f}")

        # auroc = roc_auc_score(labels, all_scores)

        # fpr, tpr, thresholds = roc_curve(labels, all_scores)
        # idx = np.where(tpr >= 0.95)[0][0]
        # fpr95 = fpr[idx]

    test_loss = test_loss / len(test_dataloader)
    test_results = metric.compute(average='weighted')

    node.model.cpu()
    return test_loss, test_results
    # return {
        # "test_loss": test_loss.item(),
        # "f1_score": test_results['f1'],
        # "auroc": auroc,
        # "fpr95": fpr95
    # }

class Recorder:
    def __init__(self, args):
        self.args = args

        # server metrics
        extra_columns = ['server_auroc', 'server_fpr95']
        node_columns = list(range(args.K * 2 + 1))
        self.test_acc = pd.DataFrame(columns=node_columns + extra_columns)

        self.val_acc = pd.DataFrame(columns=range(args.K * 2 + 1))
        self.current_test_acc = {k: None for k in range(args.K * 2 + 1)}

        # self.current_test_metrics = {0: {'auroc': None, 'fpr95': None}}
        self.current_acc = {k: None for k in range(args.K * 2 + 1)}
        self.best_acc = torch.zeros(self.args.K + 1)
        self.get_a_better = torch.zeros(self.args.K + 1)

    def evaluate(self, node, validation_dataset):
        val_loss, val_results = validation(node, validation_dataset)
        logging.info('val_loss={}, val_results={}'.format(val_loss, val_results))

        self.current_acc[node.id] = '{:.1f}'.format(val_results['f1'] * 100)
        return val_results['f1']

    def predict(self, node, test_dataset):
        # output_eval_dir = os.path.join(self.args.submission_dir, '{}_{}'.format(node.name, node.model_type))
        # os.makedirs(output_eval_dir, exist_ok=True)
        test_loss, test_results = test(node, test_dataset)
        logging.info('test_loss={}, test_results={}'.format(test_loss, test_results))
        
        self.current_test_acc[node.id] = '{:.1f}'.format(test_results['f1'] * 100)

    # def predict(self, node, test_dataset, id_test_data=None):
        # results = test(node, test_dataset, id_test_data)
        # logging.info('test_loss={}, test_results={}, auroc={}, fpr95={}'.format(results['test_loss'],
                                                            # results['f1_score'], results['auroc'], results['fpr95']))

        # self.current_test_acc[node.id] = '{:.1f}'.format(results['f1_score'] * 100)
        # #Store AUROC and FPR95 values only on the server.
        # if node.id == 0:
            # self.current_test_metrics[0]['auroc'] = '{:.2f}'.format(results['auroc'] * 100)
            # self.current_test_metrics[0]['fpr95'] = '{:.2f}'.format(results['fpr95'] * 100)

    def save_model(self, node):
        model_to_save = node.model.module if hasattr(node.model, 'module') else node.model
        model_type = node.model_type if "/" not in node.model_type else node.model_type.split("/")[-1]
        file_name = os.path.join(self.args.model_dir, '{}_{}'.format(node.name, model_type))
        model_to_save.save_pretrained(file_name)
        node.tokenizer.save_pretrained(file_name)

    def save_record(self, node=None, row=None, col=None, round_=None):   
        # centralized_mixed algorithm record the values
        if self.args.algorithm == 'centralized_mixed':
            self.val_acc.loc[len(self.val_acc)] = self.current_acc
            print(f'validation values: \n {self.val_acc}')
            self.val_acc.to_csv(os.path.join(self.args.record_dir, '{}_dev.csv'.format(self.args.algorithm)))
            # record the test values
            if self.args.do_test:
                self.test_acc.loc[len(self.test_acc)] = self.current_test_acc
                print(f'test values: \n {self.test_acc}')
                self.test_acc.to_csv(os.path.join(self.args.submission_dir, '{}_test.csv'.format(self.args.algorithm)))
        else:
            # calculate the value of row of fl algorithm
            if round_ is not None:
                row_dev = row
                row_dev += round_ * self.args.dis_epochs
                self.val_acc.at[row_dev, col] = self.current_acc[node.id]
            else:
                self.val_acc.at[row, col] = self.current_acc[node.id]
            print(f'validation values: \n {self.val_acc}')
            self.val_acc.to_csv(os.path.join(self.args.record_dir, '{}_dev.csv'.format(self.args.algorithm)))
            # record the test values
            if self.args.do_test:
                row_test_idx = row if round_ is None else row + round_ * self.args.dis_epochs
                # 1. f1 scores
                self.test_acc.at[row_test_idx, col] = self.current_test_acc[node.id]
                # #2. auroc and fpr95
                # if node.id == 0:
                    # self.test_acc.at[row_test_idx, 'server_auroc'] = self.current_test_metrics[0]['auroc']
                    # self.test_acc.at[row_test_idx, 'server_fpr95'] = self.current_test_metrics[0]['fpr95']

                print(f'test values: \n {self.test_acc}')
                self.test_acc.to_csv(
                    os.path.join(self.args.submission_dir, '{}_test.csv'.format(self.args.algorithm)))
        