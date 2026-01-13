import logging
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader


def compute_centroid(embeddings: torch.Tensor) -> torch.Tensor:
    """ Calculate the center vector (mean value) of the embedding of the sample """
    return embeddings.mean(dim=0, keepdim=True)


def get_embeddings(node, dataset):
    node.model.to(node.device).eval()

    all_embs = []
    dataset = preprocessing_raw_datasets(dataset, node.tokenizer, node.max_seq_length)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=node.batch_size, collate_fn=node.data_collator)

    for batch in tqdm(dataloader, desc='Getting Embedding'):
        batch = {k: v.to(node.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = node.model(**batch, output_hidden_states=True)
            cls_emb = outputs.hidden_states[-1][:, 0, :]
            all_embs.append(cls_emb.cpu())
    return torch.cat(all_embs, dim=0)  # shape [N, hidden_dim]


def select_topk_by_similarity(centroid, public_embs, top_k=None):
    """ Sort by cosine similarity to centroid and select the top k samples """
    sims = cosine_similarity(centroid.reshape(1, -1), public_embs).flatten()
    if top_k is None:
        return np.arange(len(sims)), sims
    top_idx = np.argsort(sims)[::-1][:top_k].tolist()

    threshold = 1e-4
    sims_topk = sims[top_idx]
    zero_count = np.sum(np.abs(sims_topk) < threshold)
    negative_count = np.sum(sims_topk < 0)
    
    return top_idx, dict(zip(top_idx, sims[top_idx]))


def weighted_kl_loss(student_logits, soft_labels, weights):
    student_probs = F.log_softmax(student_logits, dim=-1)
    loss_per_sample = F.kl_div(student_probs, soft_labels, reduction="none").sum(dim=-1)
    loss = (weights * loss_per_sample).sum() / weights.sum()
    return loss


class IgnoreSpecificMessageFilter(logging.Filter):
    def filter(self, record):
        if 'Loading cached processed dataset' in record.getMessage():
            return False
        return True


def init_model(name, model_type, num_classes):
    # config = AutoConfig.from_pretrained(model_type, num_labels=num_classes, finetuning_task=task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    total_params = total_params / 1000000
    logging.info('model parameters of %s_%s: %2.1fM' % (name, model_type.split("/")[-1], total_params))
    return tokenizer, model


def init_optimizer(optimizer_type, model, lr, weight_decay=0., momentum=0.9):
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(momentum, 0.999),
                                     eps=1e-8)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(momentum, 0.999),
                                      eps=1e-8)
    else:
        sys.exit("Not implemented optimizer, code exit, re-run to use correct optimizer")
    return optimizer


def init_scheduler(scheduler_type, optimizer, num_warmup_steps=None, num_training_steps=None):
    if scheduler_type == 'linear':
        scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                  num_training_steps=num_training_steps)
    elif scheduler_type == "cosine":  # cosine
        scheduler = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                  num_training_steps=num_training_steps)
    else:
        sys.exit("Not implemented learning rate scheduler, code exit, re-run to use correct scheduler")
    return scheduler


def preprocessing_raw_datasets(raw_dataset, tokenizer, max_seq_length, logits=None):
    text_key = 'text'

    def preprocess_function(text):
        return tokenizer(text[text_key], padding=True, max_length=max_seq_length, truncation=True)

    # define a filter fun to remove blank lines
    def filter_empty_rows(example):
        return example[text_key] is not None and example[text_key].strip() != ''

    # Apply a filter function to each data segmentation
    raw_dataset = raw_dataset.filter(filter_empty_rows)
    encoded_dataset = raw_dataset.map(preprocess_function, batched=True)
    # The cause of the error is that the tokenized datasets object has columns with strings, and the data collator does not know how to pad these
    encoded_dataset = encoded_dataset.remove_columns(text_key)
    if 'label' in encoded_dataset.column_names:
        encoded_dataset = encoded_dataset.rename_column('label', 'labels')

    if logits is not None:
        if type(logits) == list:
            for k in range(len(logits)):
                encoded_dataset = encoded_dataset.add_column('logits{}'.format(k), logits[k].tolist())
        else:
            encoded_dataset = encoded_dataset.add_column('logits', logits.tolist())

    return encoded_dataset


if __name__ == '__main__':
    raw_datasets = load_dataset('./data/yr')['train']
    print(raw_datasets)

    # Loading BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # def tokenize_function(example):
    #     return tokenizer(example['text'],padding=True, max_length=128, truncation=True)
    #
    #
    # tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    # tokenized_datasets = tokenized_datasets.remove_columns('text')
    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # tokenized_datasets.set_format("torch")
    tokenized_datasets = preprocessing_raw_datasets(raw_datasets, tokenizer, 128)
    print(tokenized_datasets)
    for i in range(5):
        print(f"Example {i}:")
        print(f"Input IDs: {tokenized_datasets[i]['input_ids']}")
        print(f"Attention Mask: {tokenized_datasets[i]['attention_mask']}")
