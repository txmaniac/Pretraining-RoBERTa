from tkinter.tix import Tree
from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM, 
    RobertaConfig, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

from datasets import load_dataset
import time
from utils import read_dataset
import os
import sys

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
config = RobertaConfig.from_pretrained('roberta-base')

model = RobertaForMaskedLM.from_pretrained('roberta-base', config=config)
dataset = load_dataset('text', data_files='requirements.txt', split='train', streaming=True)

def preprocess_function(examples):
    sentences = [q.strip() for q in examples['text']]
    
    inputs = tokenizer(
        sentences,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )

    inputs['labels'] = inputs['input_ids']
    
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns('text')
tokenized_dataset = tokenized_dataset.with_format('torch')

print(next(iter(tokenized_dataset)))