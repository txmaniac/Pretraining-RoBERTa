from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)

import sys
import torch
from utils import (
    exact_score, 
    f1_score, 
    logits_to_index, 
    presence_score, 
    read_dataset_policyqa
)
from torch.utils.data import DataLoader

if __name__ == "__main__":

    list_of_args = sys.argv[1:]
    model_path = list_of_args[0]
    dataset_path = list_of_args[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)

    correct_answers, dataset = read_dataset_policyqa(dataset_path, model_path, device)
    dataloader = DataLoader(dataset['train'], shuffle=False, batch_size=1)

    model_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)
            
            start_logits = torch.topk(outputs.start_logits.view(-1), k=3).indices
            end_logits = torch.topk(outputs.end_logits.view(-1), k=3).indices

            ordered_pairs = logits_to_index(start_logits, end_logits)

            pred_answers = []
            for x, y in ordered_pairs:
                start = x
                end = y
                pred_answers.append(tokenizer.decode(input_ids[0][start:end+1]))

            model_predictions.append(pred_answers)
    
    presence_score(model_predictions, correct_answers)
    exact_score(model_predictions, correct_answers)
    f1_score(model_predictions, correct_answers)