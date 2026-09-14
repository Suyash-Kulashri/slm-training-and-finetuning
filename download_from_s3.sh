#!/usr/bin/env bash
set -euo pipefail

S3_PATH="s3://hsihsak/SFT_covid.jsonl"
LOCAL_PATH="SFT_covid.jsonl"

echo "Downloading ${S3_PATH} to ${LOCAL_PATH} ..."

# Check that aws CLI is installed
if ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws CLI not found. Install it with:"
  echo "  pip install awscli"
  echo "or follow the AWS docs to install the AWS CLI."
  exit 1
fi

# Optional: show current AWS identity (helps debug credentials)
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "Error: AWS credentials not configured or invalid."
  echo "Run: aws configure"
  echo "and provide your AWS Access Key, Secret, region, and output format."
  exit 1
fi

# Download the file
aws s3 cp "${S3_PATH}" "${LOCAL_PATH}"

echo "Done. File saved as ${LOCAL_PATH}"