from encodings import utf_8
from datasets import load_dataset
import torch
from tqdm import tqdm
# from transformers
import os
import sys
import string
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
from transformers import AutoTokenizer

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      punct_list = list(string.punctuation)
      punct_list.remove('.')
      exclude = set(punct_list)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return remove_punc(white_space_fix(lower(s)))


def extract_sentences(path, model_path):
    list_of_folders = os.listdir(path)
    list_of_sentences = []

    for folder in tqdm(list_of_folders, desc='Converting to sentences'):
        folder_path = os.path.join(path, folder)
        list_of_files = os.listdir(folder_path)
        for file in list_of_files:
            with open(os.path.join(folder_path  ,file),encoding="utf-8") as f:
                text = f.read()
                text = normalize_answer(text)
                list_of_sentences += sent_tokenize(text)

    return list_of_sentences

def read_dataset_pretraining(path, model_path):
    # takes dataset directory path and fetches all the contents of each and every txt file and stores them as a dataset object from HuggingFace
    
    dataset = load_dataset('text', data_files=path, split='train', streaming=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess_function(examples):
        sentences = [q.strip() for q in examples['text']]
        
        inputs = tokenizer(
            sentences,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        inputs['labels'] = inputs['input_ids']
        
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns('text')
    tokenized_torch_dataset = tokenized_dataset.with_format("torch")
    

    return tokenized_torch_dataset

def read_dataset_policyqa(dataset_path, model_path, device):
        # takes dataset directory path and fetches all the contents of each and every txt file and stores them as a dataset object from HuggingFace
    
    dataset = load_dataset('json', data_files=dataset_path, field='data')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"]
            end_char = answer["answer_start"] + len(answer["text"])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs['start_positions'] = start_positions
        inputs['end_positions'] = end_positions
        
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_torch_dataset = tokenized_dataset.with_format("torch")
    
    return tokenized_torch_dataset