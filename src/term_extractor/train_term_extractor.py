# train_term_extractor.py
import os
import torch
from transformers import BigBirdTokenizerFast, BigBirdForTokenClassification, TrainingArguments, Trainer
from datasets import load_metric, Dataset
import numpy as np
from torchcrf import CRF

# NOTE: This is a simplified wrapper — production: implement CRF integrated model class

def prepare_dataset(samples):
    # samples: list of dict {"text":..., "labels": [...]} with BIO tags for tokens
    # Use tokenizer to align labels to tokens
    tokenizer = BigBirdTokenizerFast.from_pretrained("google/bigbird-roberta-base")
    tokenized_inputs = tokenizer([s["text"] for s in samples], padding='max_length', truncation=True, max_length=4096, return_offsets_mapping=True)
    # align labels (omitted for brevity)
    return tokenized_inputs

# For a fast POC, we fine-tune BigBird without CRF (paper used CRF; you can add CRF later)
def train(samples):
    tokenizer = BigBirdTokenizerFast.from_pretrained("google/bigbird-roberta-base")
    labels_list = ["O","B-TERM","I-TERM","B-DEF","I-DEF"]
    label2id = {l:i for i,l in enumerate(labels_list)}
    num_labels = len(labels_list)

    # Build toy dataset -> convert samples into token-level labels
    # Here you should implement label alignment. For brevity we assume already token-aligned.
    dataset = Dataset.from_list(samples)  # you must have tokens and labels

    model = BigBirdForTokenClassification.from_pretrained("google/bigbird-roberta-base", num_labels=num_labels)

    args = TrainingArguments(
        output_dir="./term_extractor_out",
        evaluation_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, eval_dataset=dataset)
    trainer.train()
    trainer.save_model("./term_extractor_out/model")

if __name__ == "__main__":
    # load / build your BIO-tokenized training samples here
    samples = []  # load from data/annotations
    train(samples)
