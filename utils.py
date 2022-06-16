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
from transformers import AutoTokenizer
from numpy import mean
import collections


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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    
    correct_ans = []

    for i in range(0, len(dataset['train'])):
        correct_ans.append(dataset['train'][i]['answers']['text'])

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
        answer_texts = []
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            answer_texts.append(answer['text'])
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
    
    return correct_ans, tokenized_torch_dataset

def logits_to_index(start_logits, end_logits):
  # select longest answer and return all possible ordered pairs
  max_answer_length = 1000

  ordered_pairs = []

  for start_index in start_logits:
    for end_index in end_logits:
      if end_index == start_index:
        continue

      if end_index < start_index:
        continue

      length = end_index.item() - start_index.item() + 1

      if length <= max_answer_length:
            ordered_pairs.append((start_index.item(), end_index.item()))

      else:
        continue
  
  return ordered_pairs

def presence_score(predictions, correct_ans):
    
    def compute_presence(a_gold, a_pred):
        if normalize_answer(a_pred).find(normalize_answer(a_gold)) == -1 and normalize_answer(a_gold).find(normalize_answer(a_pred)) == -1:
            return 0
        else:
            return 1
    
    presence_match = []
    best_indexes = []

    for p,c in zip(predictions, correct_ans):
        presence = 0
        index = 0
        for preds in p:
            if presence < compute_presence(c,preds):
                presence = compute_presence(c,preds)
                index = p.index(preds)
        presence_match.append(presence)
        best_indexes.append(index)

    presence_score = float(mean(presence_match)*100)
    print(f'presence score: {presence_score}')

def exact_score(predictions, correct_ans):
    
    def compute_exact(a_gold, a_pred):
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))

    exact_match = []

    for p,c in zip(predictions, correct_ans):
        x = 0
        for pred in p:
            if x < compute_exact(c,pred):
                x = compute_exact(c,pred)

        exact_match.append(x)

    exact_score = float(mean(exact_match)*100)
    print(f'exact score: {exact_score}')

def f1_score(predictions, correct_ans):

    def get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    def compute_precision(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)

        return precision

    def compute_recall(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        recall = 1.0 * num_same / len(gold_toks)

        return recall

    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    precision_list = []
    recall_list = []
    f1 = []
    best_indexes = []

    for p,c in zip(predictions, correct_ans):
        precision = 0
        recall = 0
        f1_score = 0
        index = 0
        for preds in p:
            if precision < compute_precision(c,preds):
                precision = compute_precision(c,preds)
                
            if recall < compute_recall(c,preds):
                recall = compute_recall(c,preds)
                
            if f1_score < compute_f1(c,preds):
                f1_score = compute_f1(c,preds)
                index = p.index(preds)

        #print(f'{precision} {recall} {f1_score}')
        precision_list.append(precision)
        recall_list.append(recall)
        f1.append(f1_score)
        best_indexes.append(index)

    f1_score = float(mean(f1)*100)
    print(f'Precision Score : {mean(precision_list)*100}')
    print(f'Recall Score : {mean(recall_list)*100}')
    print(f'F1 Score : {f1_score}')