from itertools import islice
import os

from datasets import Dataset, load_dataset
import argparse
import json


def build_jsonl_file (output_file: str, ds: Dataset):
  # Create output directories if they don't exist
  output_dir = os.path.dirname(os.path.abspath(output_file))
  os.makedirs(output_dir, exist_ok=True)

  count = 0
  with open(output_file, "w", encoding="utf-8") as f:
    for example in ds:
      record = {"db": example['db'],
                "schema": example['db_schema'],
                "question": example['question'], 
                "output": example['query']}
      f.write(json.dumps(record, ensure_ascii=False) + "\n")
      count += 1

  print(f"Successfully saved {count} samples to {output_file}")


def main():
  parser = argparse.ArgumentParser(description="Convert the Spider dataset to JSONL format.")
  parser.add_argument("--output_train_file", type=str, required=True, help="Path to the output train JSONL file.")
  parser.add_argument("--output_test_file", type=str, required=True, help="Path to the output test JSONL file.")
  parser.add_argument("--output_eval_file", type=str, required=True, help="Path to the output evaluation JSONL file.")    
  parser.add_argument("--num_train_samples", type=int, required=True,
                      help="Number of samples to include in the training dataset.")
  parser.add_argument("--num_test_samples", type=int, required=True,
                      help="Number of samples to include in the testing dataset.")
  parser.add_argument("--num_eval_samples", type=int, required=True,
                      help="Number of samples to include in the evaluation dataset.")
  parser.add_argument("--seed", type=int, default=321, help="Random seed for reproducible shuffling.")
  args = parser.parse_args()
  if args.num_train_samples <= 0:
    parser.error("--num_train_samples must be greater than 0")
  if args.num_test_samples <= 0:
    parser.error("--num_test_samples must be greater than 0")
  if args.num_eval_samples <= 0:
    parser.error("--num_eval_samples must be greater than 0")

  dataset = load_dataset("Turbular/fixed_spider", split="train")

  total_samples = len(dataset)
  if args.num_train_samples + args.num_test_samples + args.num_eval_samples > total_samples:
    parser.error(f"Requested {args.num_train_samples} + {args.num_test_samples} + {args.num_eval_samples} samples, "
                 f"but dataset contains only {total_samples}.")
  # Shuffle the entire dataset
  dataset = dataset.shuffle(seed=args.seed)
  dataset = dataset.select(range(args.num_train_samples + args.num_test_samples + args.num_eval_samples))

  build_jsonl_file(args.output_train_file, dataset.select(range(args.num_train_samples)))
  build_jsonl_file(args.output_test_file, 
                   dataset.select(range(args.num_train_samples, args.num_train_samples + args.num_test_samples)))
  build_jsonl_file(args.output_eval_file, 
                   dataset.select(range(len(dataset) - args.num_eval_samples, len(dataset))))


if __name__ == "__main__":
    main()