import pandas as pd
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments
model_name = "google/mt5-large"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

import json

test_set = {}
test= []

with open("./spanish_prompt.jsonl", "r") as f:
  for line in f:
    test_set.update(json.loads(line))
    test.append(json.loads(line))
total = len(test_set)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
    overwrite_output_dir=True,
    logging_dir="./logs",
    max_steps=16,
)

num_train = int(total * .8)
num_val = int(total * .1)
num_test = total - num_train - num_val
texts = []
for i in range(total):
  texts.append(test[i]['prompt'])
train_text = [texts[i] for i in range(num_train)]
val_text = [texts[i] for i in range(num_train, num_val+num_train)]
test_text = [texts[i] for i in range(num_val + num_train, total)]

train_dataset = pd.DataFrame(train_text)
val_data = pd.DataFrame(val_text)
test_data = pd.DataFrame(test_text)

def preprocess_function(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True)
df = pd.DataFrame(test)

train_dataset = test.with_format("torch")
dataset = train_dataset.with_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])

# Apply preprocessing to the dataset
train_dataset = map(preprocess_function, test)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(model=model,args=training_args,train_dataset=train_dataset,compute_metrics=compute_metrics)

# Fine-tune the model
trainer.train()

# Evaluate the fine-tuned model
results = trainer.evaluate()
print(results)