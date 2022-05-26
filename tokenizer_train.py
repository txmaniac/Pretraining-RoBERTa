import os
from tqdm import tqdm
import sys
from transformers import AutoTokenizer
from datasets import load_dataset

if __name__ == "__main__":
    list_of_args = sys.argv[1:]

    path = list_of_args[0]
    model_path = list_of_args[1]
    
    # paths = []

    # for i in tqdm(range(1,102), desc='Reading directories'):
    #     path = os.path.join(dataset_path, str(i))
    #     list_of_files = os.listdir(path)

    #     for file in list_of_files:
    #         paths.append(os.path.join(path, file))

    dataset = load_dataset('text', data_files=path, split='train')

    old_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    new_tokenizer = old_tokenizer.train_new_from_iterator(dataset, 30000)
    #Save the Tokenizer to disk
    new_tokenizer.save_pretrained(model_path)