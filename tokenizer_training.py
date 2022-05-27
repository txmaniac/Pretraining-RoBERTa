from transformers import RobertaForMaskedLM, RobertaTokenizerFast
from datasets import load_dataset
import sys

data_path = sys.argv[1]
model_path = sys.argv[2]

old_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

dataset = load_dataset('text', data_files=data_path, split='train')

batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]['text']


new_tokenizer = old_tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=30000)

model.save_pretrained(model_path)
new_tokenizer.save_pretrained(model_path)