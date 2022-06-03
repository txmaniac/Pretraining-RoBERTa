from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM, 
    RobertaConfig, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    RobertaTokenizer
)

from tokenizers import Tokenizer

import time
from utils import read_dataset
import os
import sys

if __name__ == "__main__":

    train_dir_path = sys.argv[1]
    eval_dir_path = sys.argv[2]
    model_path = sys.argv[3]
    tokenizer_path = sys.argv[4]
    logging_path = sys.argv[5]
    output_path = sys.argv[6]
    resume_chckpt = int(sys.argv[7])
    # resume_path = sys.argv[7]

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    config = RobertaConfig.from_pretrained(model_path)

    model = RobertaForMaskedLM.from_pretrained(model_path, config=config)

    print('Loading dataset...')
    train_dataset = read_dataset(train_dir_path, model_path)
    eval_dataset = read_dataset(eval_dir_path, model_path)

    print('Loading Collator...')
    train_batch_size = 10
    eval_batch_size = 10
    
    data_collator = DataCollatorForLanguageModeling(
        mlm=True,
        tokenizer=tokenizer,
        mlm_probability=0.1
    )

    print('Setting training args...')
    training_args = TrainingArguments(
        do_train=True,
        output_dir = output_path,
        evaluation_strategy="steps",
        logging_steps=1000,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=1e-4,
        warmup_steps=300,
        eval_steps = 1000,
        save_steps = 50000,
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.01,
        max_steps=2000000,
        logging_dir=logging_path
    )

    print('Preparing Trainer...')
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )

    print('Training starts...')
    start = time.time()
    
    if resume_chckpt == 1:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    end = time.time()

    print(f'Training completed in {end-start}')