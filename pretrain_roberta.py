from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM, 
    RobertaConfig, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

import time
from utils import read_dataset
import os
import sys

if __name__ == "__main__":

    dir_path = sys.argv[1]
    model_path = sys.argv[2]
    logging_path = sys.argv[3]
    output_path = sys.argv[4]

    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    config = RobertaConfig.from_pretrained(model_path)

    model = RobertaForMaskedLM.from_pretrained(model_path, config=config)

    train_dataset = read_dataset(dir_path, model_path)

    data_collator = DataCollatorForLanguageModeling(
        mlm=True,
        tokenizer=tokenizer,
        mlm_probability=0.1
    )

    print('Setting training args...')
    training_args = TrainingArguments(
        do_train=True,
        output_dir = output_path,
        per_device_train_batch_size=16,
        learning_rate=6e-4,
        warmup_steps=30000,
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.01,
        max_steps=500000,
        logging_dir=logging_path
    )

    print('Preparing Trainer...')
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset
    )

    print('Training starts...')
    start = time.time()
    trainer.train()
    end = time.time()

    print(f'Training completed in {end-start}')