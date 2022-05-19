from encodings import utf_8
from datasets import Dataset
import torch
from tqdm import tqdm
# from transformers
import os
import sys
import string
import nltk
nltk.download('punkt')
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

def read_dataset(path, model_path):
    # takes dataset directory path and fetches all the contents of each and every txt file and stores them as a dataset object from HuggingFace
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    list_of_folders = os.listdir(path)
    list_of_sentences = []
    i=0
    for folder in tqdm(list_of_folders, desc='Converting policies to sentences',position=0, leave=True):
        
        if i == 5:
            break
        else:
            i+=1
            folder_path = os.path.join(path, folder)
            list_of_files = os.listdir(folder_path)
            for file in list_of_files:
                with open(os.path.join(folder_path  ,file),encoding="utf-8") as f:
                    text = f.read()
                    text = normalize_answer(text)
                    list_of_sentences += sent_tokenize(text)

    data_dict = {'sentences': list_of_sentences, 'labels': tokenizer(list_of_sentences, truncation=True, padding=True)['input_ids']}

    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.map(lambda examples: tokenizer(examples['sentences'], truncation=True, padding=True), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return dataset