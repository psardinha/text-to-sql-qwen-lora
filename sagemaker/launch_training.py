import subprocess
from pathlib import Path

import boto3

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import CheckpointConfig, Compute, InputData, OutputDataConfig, SourceCode, StoppingCondition
from sagemaker.train import ModelTrainer


ROOT = Path(__file__).resolve().parent
INFRASTRUCTURE_DIR = ROOT.parent / "infrastructure"

REGION = "eu-west-1"
INSTANCE_TYPE = "ml.g5.xlarge"

IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/huggingface-pytorch-training:2.11.0-transformers5.13.1-cu130-amzn2023"

TRAIN_FILE = ROOT.parent / "data" / "train.jsonl"
EVAL_FILE = ROOT.parent / "data" / "eval.jsonl"


def terraform_output(name):
    return subprocess.check_output(["terraform", f"-chdir={INFRASTRUCTURE_DIR}", "output", "-raw", name], text=True).strip()


# Check input files.
for file in (TRAIN_FILE, EVAL_FILE):
  if not file.exists():
    raise FileNotFoundError(file)

# Get Terraform outputs.
BUCKET = terraform_output("s3_bucket_name")
ROLE_ARN = terraform_output("sagemaker_role_arn")
try:
  WANDB_SECRET_ARN = terraform_output("wandb_secret_arn")
except subprocess.CalledProcessError:
  WANDB_SECRET_ARN = ""

environment = {"AWS_DEFAULT_REGION": REGION,
               "WANDB_PROJECT"     : "qwen-text-to-sql"}
if WANDB_SECRET_ARN:
  environment["WANDB_SECRET_ARN"] = WANDB_SECRET_ARN


# AWS session.
boto_session = boto3.Session(region_name=REGION)
s3 = boto_session.client("s3")

# Upload training and evaluation data.
s3.upload_file(str(TRAIN_FILE), BUCKET, "data/train/train.jsonl")
s3.upload_file(str(EVAL_FILE), BUCKET, "data/eval/eval.jsonl")

train_uri = f"s3://{BUCKET}/data/train/"
eval_uri = f"s3://{BUCKET}/data/eval/"

# SageMaker v3 session.
sm_session = Session(boto_session=boto_session, default_bucket=BUCKET)

# Training configuration.
trainer = ModelTrainer(role=ROLE_ARN,
                       sagemaker_session=sm_session,

                       # AWS-managed Hugging Face container.
                       training_image=IMAGE_URI,

                       # Training script.
                       source_code=SourceCode(source_dir=str(ROOT), entry_script="run_perf.py", requirements="requirements.txt"),

                       # Compute resources.
                       compute=Compute(instance_type=INSTANCE_TYPE, instance_count=1,  enable_managed_spot_training=True),

                       # Training time limits.
                       stopping_condition=StoppingCondition(max_runtime_in_seconds=2 * 60 * 60, max_wait_time_in_seconds=3 * 60 * 60),

                       # Checkpoints.
                       checkpoint_config=CheckpointConfig(s3_uri=f"s3://{BUCKET}/checkpoints/", local_path="/opt/ml/checkpoints"),

                       # Model output.
                       output_data_config=OutputDataConfig(s3_output_path=f"s3://{BUCKET}/model-artifacts/"),

                       # Environment variables.
                       environment= environment,

                       # Training hyperparameters.
                       hyperparameters={"model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                                        "max_length": 2048,
                                        "num_train_epochs": 2,
                                        "learning_rate": 1e-4,
                                        "per_device_train_batch_size": 2,
                                        "per_device_eval_batch_size": 2,
                                        "gradient_accumulation_steps": 4,
                                        "gradient_checkpointing": True})

print(f"Starting SageMaker training on {INSTANCE_TYPE} in {REGION}")

# Start training.
trainer.train(input_data_config=[InputData(channel_name="train", data_source=train_uri),
                                 InputData(channel_name="eval", data_source=eval_uri)])

print("Training job completed.")
print(f"Model artifacts: s3://{BUCKET}/model-artifacts/")