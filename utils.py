from encodings import utf_8
from datasets import load_dataset
import torch
from tqdm import tqdm
# from transformers
import os
import sys
import string
# import nltk
# nltk.download('punkt')
from nltk import sent_tokenize
from transformers import RobertaTokenizer

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
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
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

def read_dataset(path, model_path):
    # takes dataset directory path and fetches all the contents of each and every txt file and stores them as a dataset object from HuggingFace
    
    dataset = load_dataset('text', data_files=path, streaming=True)

    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    def preprocess_function(examples):
        sentences = [q.strip() for q in examples['text']]
        
        inputs = tokenizer(
            sentences,
            truncation="True",
            padding="True",
        )

        inputs['labels'] = sentences
        
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, )

    return tokenized_dataset, len(tokenized_dataset)