import sys

import json
import torch
import torch.nn as nn

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Read dataset
def get_datasets(tokenizer, finetune_dataset_name):
    try:
        with open(finetune_dataset_name, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError as e:
        print(f"The file {e} was not found.")
        return None, None

    train_data = dataset['train']
    train_dataset = Dataset.from_list(train_data)

    train_test_split = train_dataset.train_test_split(test_size=0.2)

    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    def preprocess_function(examples):
        inputs = tokenizer(examples['prompt'],
                           padding='max_length',
                           truncation=True,
                           max_length=512,
                           return_tensors="pt")

        targets = tokenizer(examples['solution'],
                            padding='max_length',
                            truncation=True,
                            max_length=512,
                            return_tensors="pt")

        inputs['labels'] = targets['input_ids']

        return inputs

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['prompt', 'solution'])
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(['prompt', 'solution'])

    return tokenized_train_dataset, tokenized_test_dataset


# Fine tuning model with save in the end
def finetune_model(model_name='ibm-granite/granite-3b-code-base-2k',
                   finetune_dataset_name='test_dataset.json'):
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # freezing model layers
    for param in model.parameters():
        param.requires_grad = False

        if param.ndim == 1:
            param.data = param.data.to(torch.bfloat16)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.bfloat16)

    model.lm_head = CastOutputToFloat(model.lm_head)

    # Set up layers for finetuning
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    tokenized_train_dataset, tokenized_test_dataset = get_datasets(tokenizer, finetune_dataset_name)

    if tokenized_train_dataset is None and tokenized_test_dataset is None:
        return None

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=30,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        remove_unused_columns=False,
        fp16=True,
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Evaluating the model...")
    evaluation_results = trainer.evaluate()

    print("Evaluation results:", evaluation_results)

    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")

    print("Model fine-tuned and saved successfully.")

    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        finetune_model()
    elif len(sys.argv) == 2:
        finetune_model(sys.argv[1])
    elif len(sys.argv) == 3:
        finetune_model(sys.argv[1], sys.argv[2])
    else:
        print("Wrong number of arguments")