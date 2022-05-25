from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
from tqdm import tqdm
import sys

if __name__ == "__main__":
    list_of_args = sys.argv[1:]

    dataset_path = list_of_args[0]
    tokenizer_save_path = list_of_args[1]
    paths = []

    for i in tqdm(range(1,102), desc='Reading directories'):
        path = os.path.join(dataset_path, str(i))
        list_of_files = os.listdir(path)

        for file in list_of_files:
            paths.append(os.path.join(path, file))

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])

    tokenizer.pre_tokenizer = Whitespace()
    
    tokenizer.train(paths, trainer)
    
    #Save the Tokenizer to disk
    tokenizer.save_pretrained(os.path.join(tokenizer_save_path))