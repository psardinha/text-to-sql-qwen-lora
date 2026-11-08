import torch
import transformers
import wandb
import weave

from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


import os
from pathlib import Path
import boto3

print("Transformers version:", transformers.__version__)
if transformers.__version__ != "5.13.1":
  raise RuntimeError(f"Expected Transformers 5.13.1, got {transformers.__version__}")

os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "qwen-text-to-sql")
region = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")
secret_arn = os.environ.get("WANDB_SECRET_ARN")
if secret_arn:
  secrets_client = boto3.client("secretsmanager", region_name=region)
  response = secrets_client.get_secret_value(SecretId=secret_arn)
  os.environ["WANDB_API_KEY"] = response["SecretString"]

device_type = "cuda" if torch.cuda.is_available() else "cpu"
model_dtype = torch.float16 if device_type == "cuda" else torch.float32
print("Training device:", device_type)
print("Model dtype:", model_dtype)
if not torch.cuda.is_available():
  raise RuntimeError("CUDA is not available. Check the SageMaker GPU environment.")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 2048


# 1. Load datasets
train_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", "data"))
eval_dir = Path(os.environ.get("SM_CHANNEL_EVAL", "data"))
train_ds = load_dataset("json", data_files=str(train_dir / "train.jsonl"), split="train")
eval_ds = load_dataset("json", data_files=str(eval_dir / "eval.jsonl"), split="train")

# 1. Initialize W&B
if secret_arn:
  weave.init('pasardinha-experimenting-on-my-own/text-to-sql')
  wandb.init(project="text-to-sql",
             entity="pasardinha-experimenting-on-my-own",
             name="qwen2.5-1.5b-lora",
             config={"model": model_name,
                     "max_length": MAX_LENGTH,
                     "training_samples": len(train_ds),
                     "evaluation_samples": len(eval_ds),
                     "epochs": 2,
                     "lora_rank": 8,
                     "learning_rate": 1e-4,
                     "batch_size": 1,
                     "gradient_accumulation_steps": 8,
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
training_args = TrainingArguments(output_dir="/opt/ml/checkpoints",
                                  num_train_epochs=2,
                                  learning_rate=1e-4,
                                  per_device_train_batch_size=2,
                                  per_device_eval_batch_size=2,
                                  gradient_accumulation_steps=4,
                                  logging_strategy="steps",
                                  logging_steps=25,
                                  eval_strategy="epoch",
                                  save_strategy="epoch",
                                  save_total_limit=2,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="eval_loss",
                                  greater_is_better=False,
                                  fp16=(device_type == "cuda"),
                                  bf16=False,
                                  use_cpu=(device_type == "cpu"),
                                  gradient_checkpointing=True,
                                  report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
                                  run_name="qwen2.5-1.5b-lora-sagemaker")

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

  # Create labels before truncation.
  labels = full_ids.copy()

  # Ignore prompt tokens during training.
  prompt_length = min(len(prompt_ids), len(labels))
  labels[:prompt_length] = [-100] * prompt_length

  # Truncate inputs and labels consistently.
  full_ids = full_ids[:MAX_LENGTH]
  labels = labels[:MAX_LENGTH]

  # Ensure at least one SQL token remains.
  if all(label == -100 for label in labels):
    raise ValueError("No SQL response tokens remain after truncation. Increase MAX_LENGTH or shorten the prompt.")

  # Verify the complete SQL answer is present.
  if len(full_result["input_ids"]) > MAX_LENGTH:
    raise ValueError("Example exceeds MAX_LENGTH. The SQL response may be truncated. Increase MAX_LENGTH or shorten the schema/prompt.")
  
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
  checkpoint_path = Path("/opt/ml/checkpoints")
  checkpoint_path.mkdir(parents=True, exist_ok=True)
  checkpoint_dirs = list(checkpoint_path.glob("checkpoint-*"))

  if checkpoint_dirs:
      latest_checkpoint = max(checkpoint_dirs, key=lambda p: int(p.name.split("-")[-1]))
      print(f"Resuming from {latest_checkpoint}")
      trainer.train(resume_from_checkpoint=str(latest_checkpoint))
  else:
      print("Starting a new training run")
      trainer.train()

  # Save LoRA adapter
  trainer.save_model("/opt/ml/model")

  # Save tokenizer
  tokenizer.save_pretrained("/opt/ml/model")
  print("Training completed successfully!")

finally:
    wandb.finish()
