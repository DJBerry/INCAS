from transformers import CamembertTokenizer,CamembertModel
import torch
from torch.utils.data import Dataset, DataLoader
tokenizer = CamembertTokenizer.from_pretrained("./camembert-base/")

from datasets import load_dataset
datasets=load_dataset("csv", data_files={"train": "./train_terr.csv", "validation": "./val_terr.csv"})

print(datasets)

def tokenize_function(examples):
    return tokenizer(examples["cleaned_text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["cleaned_text"])

block_size=128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))

from transformers import CamembertForMaskedLM
model = CamembertForMaskedLM.from_pretrained('./camembert-base')

model_checkpoint='camembert-base'

from transformers import Trainer, TrainingArguments
model_name = model_checkpoint.split("-")[0]
training_args = TrainingArguments(
    f"{model_name}-finetuned-twitter",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_steps=10000
)

import torch
print(torch.cuda.is_available())

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.save_model('./camembert_terr_model/')
