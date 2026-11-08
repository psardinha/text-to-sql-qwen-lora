import torch
import wandb
import weave

from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


import os
from getpass import getpass

wandb_api_key = os.environ.get("WANDB_API_KEY")

if not wandb_api_key:
    wandb_api_key = getpass("Enter your W&B API key: ")
    os.environ["WANDB_API_KEY"] = wandb_api_key

# 1. Hardware detection
if torch.cuda.is_available():
  device_type = "cuda"
  if torch.cuda.is_bf16_supported():
    model_dtype = torch.bfloat16
    use_bf16 = True
    use_fp16 = False
  else:
    model_dtype = torch.float16
    use_bf16 = False
    use_fp16 = True
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
  device_type = "mps"
  model_dtype = torch.float32
  use_bf16 = False
  use_fp16 = False
else:
  device_type = "cpu"
  model_dtype = torch.float32
  use_bf16 = False
  use_fp16 = False
use_cpu = device_type == "cpu"

print(f"Detected device: {device_type}")
print(f"Model dtype: {model_dtype}")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 1024

# 1. Initialize W&B
weave.init('pasardinha-experimenting-on-my-own/text-to-sql')
wandb.init(project="text-to-sql",
           entity="pasardinha-experimenting-on-my-own",
           name="qwen2.5-1.5b-lora",
           config={"model": "Qwen2.5-1.5B-Instruct",
                   "training_samples": 2400,
                   "epochs": 1,
                   "lora_rank": 8,
                   "learning_rate": 2e-4,
                   "device": device_type})

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

# 3. Load base model
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# Save memory by enabling gradient checkpointing and disabling caching
model.gradient_checkpointing_enable()
model.config.use_cache = False

# 4. Configure LoRA
training_args = TrainingArguments(output_dir="./qwen2.5-lora",
                                  num_train_epochs=4,
                                  learning_rate=2e-4,
                                  per_device_train_batch_size=2,
                                  per_device_eval_batch_size=2,
                                  gradient_accumulation_steps=4,
                                  logging_steps=1,
                                  eval_strategy="epoch",
                                  #eval_strategy="steps",
                                  #eval_steps=1,
                                  save_strategy="epoch",
                                  load_best_model_at_end=True,
                                  metric_for_best_model="eval_loss",
                                  greater_is_better=False,
                                  fp16=(device_type == "cuda"),
                                  bf16=False,
                                  use_cpu=(device_type == "cpu"),
                                  report_to="wandb")

print("\n--- Training configuration ???---")
print("Eval strategy:", training_args.eval_strategy)
print("Save strategy:", training_args.save_strategy)
print("Load best model:", training_args.load_best_model_at_end)
print("------------------------------\n")

# 4. Configure LoRA
lora_config = LoraConfig(r=8,
                         lora_alpha=16,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         lora_dropout=0.05,
                         bias="none",
                         task_type=TaskType.CAUSAL_LM)

# 5. Attach LoRA adapters
model = get_peft_model(model, lora_config)

# Check trainable parameters
print("\n\nTrainable parameters after applying LoRA:")
model.print_trainable_parameters()

# 6. Tokenize train and eval datasets
train_ds = load_dataset("json", data_files="data/train.jsonl", split="train")
eval_ds = load_dataset("json", data_files="data/eval.jsonl", split="train")

SYSTEM_PROMPT = """You are an expert text-to-SQL assistant.
Given a database schema and a natural-language question,
generate the SQL query that answers the question.
Return only the SQL query, without explanations."""

def tokenize_function(example):
  messages = [{"role": "system",
               "content": SYSTEM_PROMPT},
              {"role": "user",
               "content": f"""Database: {example['db']}
Schema:
{example['schema']}

Question:
{example['question']}"""}]

  # Tokenize the prompt (system + user messages)
  prompt_result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)
  # Add the expected SQL response
  full_messages = messages + [{"role": "assistant", "content": example["output"]}]

  # Tokenize the complete conversation
  full_result = tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False, return_dict=True)

  # Extract token IDs
  prompt_ids = prompt_result["input_ids"]
  full_ids = full_result["input_ids"]

  # Convert tensors to Python lists if necessary
  if hasattr(prompt_ids, "tolist"):
    prompt_ids = prompt_ids.tolist()
  if hasattr(full_ids, "tolist"):
    full_ids = full_ids.tolist()

  # Remove batch dimensions if present
  if len(prompt_ids) > 0 and isinstance(prompt_ids[0], list):
    prompt_ids = prompt_ids[0]

  if len(full_ids) > 0 and isinstance(full_ids[0], list):
    full_ids = full_ids[0]

  # Ensure token IDs are ordinary Python integers
  prompt_ids = [int(token) for token in prompt_ids]
  full_ids = [int(token) for token in full_ids]

  # Truncate to maximum sequence length
  full_ids = full_ids[:MAX_LENGTH]

  # Create labels: ignore prompt tokens during training
  labels = full_ids.copy()
  prompt_length = min(len(prompt_ids), len(full_ids))
  labels[:prompt_length] = [-100] * prompt_length

  # Return data in Hugging Face Datasets-compatible format
  return {"input_ids": full_ids,
          "attention_mask": [1] * len(full_ids),
          "labels": labels}


tokenized_train_ds = train_ds.map(tokenize_function, remove_columns=train_ds.column_names)
tokenized_eval_ds = eval_ds.map(tokenize_function, remove_columns=eval_ds.column_names)

def count_empty_labels(dataset):
  return sum(1 for example in dataset if all(label == -100 for label in example["labels"]))

empty_train = count_empty_labels(tokenized_train_ds)
empty_eval = count_empty_labels(tokenized_eval_ds)
print("Training examples without SQL tokens:", empty_train)
print("Evaluation examples without SQL tokens:", empty_eval)
if empty_train > 0 or empty_eval > 0:
  raise ValueError("Some examples have no SQL tokens after truncation. Increase MAX_LENGTH or investigate those examples.")

# Dynamic padding
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt")

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=tokenized_train_ds,
                  eval_dataset=tokenized_eval_ds,
                  data_collator=data_collator,
                  processing_class=tokenizer,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0)])

# 7. Start training
try:
  # Start training
  trainer.train()

  # Save LoRA adapter
  trainer.save_model("./qwen2.5-1.5b-lora")

  # Save tokenizer
  tokenizer.save_pretrained("./qwen2.5-1.5b-lora")
  print("Training completed successfully!")

finally:
    wandb.finish()
