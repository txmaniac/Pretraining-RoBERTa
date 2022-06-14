from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM, 
    RobertaConfig, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    RobertaTokenizer
)

import time
from utils import read_dataset
import os
import sys

if __name__ == "__main__":

    train_dir_path = sys.argv[1]
    model_path = sys.argv[2]
    tokenizer_path = sys.argv[3]
    logging_path = sys.argv[4]
    output_path = sys.argv[5]
    # resume_path = sys.argv[7]

    from transformers import PreTrainedTokenizerFast

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    config = RobertaConfig.from_pretrained(model_path)

    model = RobertaForMaskedLM.from_pretrained(model_path, config=config)

    print('Loading dataset...')
    train_dataset = read_dataset(train_dir_path, model_path)

    print('Loading Collator...')
    train_batch_size = 4
    
    data_collator = DataCollatorForLanguageModeling(
        mlm=True,
        tokenizer=tokenizer,
        mlm_probability=0.1
    )

    print('Setting training args...')
    training_args = TrainingArguments(
        do_train=True,
        output_dir = output_path,
        per_device_train_batch_size=train_batch_size,
        learning_rate=1e-4,
        warmup_steps=300,
        logging_steps=100,
        save_steps = 50000,
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.01,
        max_steps=100000,
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
    trainer.train(resume_from_checkpoint=True)
    end = time.time()

    print(f'Training completed in {end-start}')