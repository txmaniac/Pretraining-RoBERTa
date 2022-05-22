from utils import extract_sentences
import sys
import os

if __name__ == "__main__":

    train_dir_path = sys.argv[1]
    eval_dir_path = sys.argv[2]
    model_path = sys.argv[3]

    train_dataset = extract_sentences(train_dir_path, model_path)
    eval_dataset = extract_sentences(eval_dir_path, model_path)

    with open(os.path.join(train_dir_path,'train.txt'), 'w') as train_file:
        train_file.writelines("%s\n" % sent for sent in train_dataset)

    with open(os.path.join(train_dir_path,'eval.txt'), 'w') as eval_file:
        eval_file.writelines("%s\n" % sent for sent in eval_dataset)