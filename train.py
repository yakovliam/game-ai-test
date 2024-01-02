# create huggingface dataset from json file
from datasets import Dataset
import json

dataset_list = []
with open("dataset_list.json") as f:
    dataset_list = json.load(f)

dataset = Dataset.from_list(dataset_list)

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["prompt"])


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset.column_names,
)

# split the dataset into 80% training and 20% validation sets
train_dataset = tokenized_dataset.shard(index=0, num_shards=5)
test_dataset = tokenized_dataset.shard(index=1, num_shards=5)

from transformers import DataCollatorForLanguageModeling

# tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from transformers import TrainingArguments, Trainer

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="proper-to-gamerspeak-model-2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

trainer.train()
