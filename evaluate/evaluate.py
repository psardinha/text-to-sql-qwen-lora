import re
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = """You are an expert text-to-SQL assistant.
Given a database schema and a natural-language question,
generate the SQL query that answers the question.
Return only the SQL query, without explanations."""

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def resolve_path(path):
  """Resolve relative paths from the project root."""
  path = Path(path)
  if path.is_absolute():
    return path
  return ROOT / path


def str_to_bool(value):
  """Convert command-line string to boolean."""

  value = value.lower()
  if value in ("true", "1", "yes"):
     return True
  if value in ("false", "0", "no"):
    return False
  raise argparse.ArgumentTypeError("Expected True or False")


def get_gold_sql(example):
  """Get reference SQL from the evaluation example."""

  for field in ("output", "expected_sql", "query"):
    sql = example.get(field)
    if isinstance(sql, str) and sql.strip():
      return sql.strip()
  raise ValueError("Gold SQL not found. Expected a non-empty 'output', 'expected_sql', or 'query' field.")


def make_single_line(sql):
  """Convert SQL into a single line for Spider files."""
  sql = re.sub(r'\s+', ' ', sql).strip()
  sql = re.sub(r';\s*$', '', sql).strip()
  return sql


# --------------------------------------------------
# Load model
# --------------------------------------------------  

def get_device_and_dtype():
  """Select the best available hardware and data type."""

  if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
      return torch.device("cuda"), torch.bfloat16
    return torch.device("cuda"), torch.float16

  if torch.backends.mps.is_available():
    return torch.device("mps"), torch.float16
  return torch.device("cpu"), torch.float32


def load_adapted_model(use_lora, adapter_dir):
  """Load the model eventually with LoRA adapter if specified."""    
  device, dtype = get_device_and_dtype()
  print(f"Device: {device}")
  print(f"Data type: {dtype}")

  print("Loading tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"
  print("Loading base model...")
  print(f"Base model: {BASE_MODEL}")

  model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
  if use_lora:
    adapter_path = resolve_path(adapter_dir)
    if not (adapter_path / "adapter_config.json").exists():
      raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
  else:
    print("Using original model without LoRA.")

  model = model.to(device)
  model.eval()
  print("Model loaded successfully!")
  return model, tokenizer, device


# --------------------------------------------------
# Generate SQL
# --------------------------------------------------

def generate_sql(model, tokenizer, example, device, max_new_tokens=2048):
  messages = [{"role": "system",
               "content": SYSTEM_PROMPT},
              {"role": "user",
               "content": f"""Database: {example['db']}

Schema:
{example['schema']}

Question:
{example['question']}"""}]

  inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

  # The model is running on CPU.
  inputs = {key: value.to(device) for key, value in inputs.items()}

  with torch.inference_mode():
    outputs = model.generate(**inputs,  max_new_tokens=max_new_tokens, do_sample=False, 
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

  # Remove the input prompt from the generated sequence.
  prompt_length = inputs["input_ids"].shape[1]
  generated_tokens = outputs[0][prompt_length:]
  prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
  return prediction.strip()


# --------------------------------------------------
# Evaluate and generate Spider files
# --------------------------------------------------

def evaluate(model, tokenizer, device, trim, eval_file, output_file, gold_file, pred_file, max_new_tokens):
  if not eval_file.exists():
    raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

  # Create output directories
  for path in (output_file, gold_file, pred_file):
    path.parent.mkdir(parents=True, exist_ok=True)
  # Count non-empty examples
  with eval_file.open("r", encoding="utf-8") as f:
    total = sum(1 for line in f if line.strip())
  print(f"\nEvaluation examples: {total}")
  print("Starting inference...\n")

  processed = 0

  with (eval_file.open("r", encoding="utf-8") as input_f,
        output_file.open("w", encoding="utf-8") as json_f,
        gold_file.open("w", encoding="utf-8") as gold_f,
        pred_file.open("w", encoding="utf-8") as pred_f):
    for line_number, line in enumerate(input_f, start=1):
      if not line.strip():
        continue
      example = json.loads(line)

      # Validate required fields
      for field in ("db", "schema", "question"):
        if field not in example:
          raise ValueError(f"Missing '{field}' in input line {line_number}")

      db_name = example["db"]
      if not isinstance(db_name, str) or not db_name.strip():
        raise ValueError(f"Invalid database ID in line {line_number}")
      gold_sql = get_gold_sql(example)

      # Generate prediction
      prediction = generate_sql(model, tokenizer, example, device, max_new_tokens=max_new_tokens)
      if trim:
        # Remove ```sql prefix and ``` suffix if present
        if prediction.startswith("```sql"):
          prediction = prediction[len("```sql"):].lstrip()
          if prediction.endswith("```"):
            prediction = prediction[:-len("```")].rstrip()

      # Save JSONL output
      result = {**example, "prediction": prediction}

      json_f.write(json.dumps(result, ensure_ascii=False) + "\n")
      json_f.flush()

      # Spider gold format:
      # GOLD_SQL<TAB>DATABASE_ID
      gold_f.write(f"{make_single_line(gold_sql)}\t{db_name}\n")
      gold_f.flush()

      # Spider prediction format:
      # One predicted SQL query per line
      pred_f.write(make_single_line(prediction) + "\n")
      pred_f.flush()

      processed += 1
      print(f"[{processed}/{total}] Database: {db_name}")
      print(f"Prediction: {prediction}")
      print("-" * 60)

  # Final consistency check
  if processed != total:
    raise RuntimeError(f"Expected {total} examples, processed {processed}.")
  print("\nEvaluation completed!")
  print(f"Examples processed: {processed}")
  print(f"JSONL output      : {output_file}")
  print(f"Spider gold file  : {gold_file}")
  print(f"Spider pred file  : {pred_file}")



# Main
def main():
  parser = argparse.ArgumentParser(description="Run CPU inference with a Qwen and optionally LoRA adapter.")
  parser.add_argument("--use-lora", type=str_to_bool, default="True", help="False for original model; True for LoRA adapter.")
  parser.add_argument("--trim", type=str_to_bool, default="False", 
                      help="If True, remove ```sql prefix and ``` suffix from predictions.")
  parser.add_argument("--adapter-dir", type=str, default="qwen2.5-1.5b-lora", help="Directory containing the trained LoRA adapter.")
  parser.add_argument("--eval-file", type=str, default="data/eval2.jsonl", help="Evaluation JSONL file.")
  parser.add_argument("--output-file", type=str, default="output/eval_predictions.jsonl", help="Output JSONL file.")
  parser.add_argument("--gold-file", type=str, default="output/gold.sql", help="REference SQL file in Spider format.")
  parser.add_argument("--pred-file", type=str, default="output/predictions.txt", help="Predicted SQL file in Spider format.")
  parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum number of tokens to generate per example.")
  args = parser.parse_args()

  if args.max_new_tokens <= 0:
    parser.error("--max-new-tokens must be positive")

  # Resolve paths relative to project root
  eval_file = resolve_path(args.eval_file)
  output_file = resolve_path(args.output_file)
  gold_file = resolve_path(args.gold_file)
  pred_file = resolve_path(args.pred_file)

  # Load model and tokenizer
  model, tokenizer, device = load_adapted_model(use_lora=args.use_lora, adapter_dir=args.adapter_dir)

  print(f"Device: {str(device)}")
  print(f"PyTorch version: {torch.__version__}")
  print(f"Using LoRA: {args.use_lora}")
  
  # Run evaluation
  evaluate(model=model, tokenizer=tokenizer, device=device, trim=args.trim, eval_file=eval_file, output_file=output_file, 
           gold_file=gold_file, pred_file=pred_file, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
  main()
