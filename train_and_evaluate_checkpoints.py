from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    DefaultDataCollator,
    TrainingArguments,
)

from utils import read_dataset_policyqa
import torch

if __name__ == "__main__":
    model_path = "mukund/privbert"
    
    data_collator = DefaultDataCollator()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
    
    train_dataset_path = 'converted_train.json'
    eval_dataset_path = 'converted_dev.json'

    train_dataset = read_dataset_policyqa(train_dataset_path, model_path, device)
    eval_dataset = read_dataset_policyqa(eval_dataset_path, model_path, device)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy='epoch',
        save_steps=5000,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model = model,
        args=training_args,
        train_dataset=train_dataset['train'],
        eval_dataset=eval_dataset['train'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.evaluate()