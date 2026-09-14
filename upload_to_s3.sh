#!/usr/bin/env bash
set -euo pipefail

BUCKET="s3://hsihsak"
LOCAL_FILE="llama3.1-8b-covid-sft-f16.gguf"
S3_DEST="${BUCKET}/${LOCAL_FILE}"

echo "Uploading ${LOCAL_FILE} to ${S3_DEST} ..."

# Check that aws CLI is installed
if ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws CLI not found. Install it with:"
  echo "  pip install awscli"
  echo "or follow the AWS docs to install the AWS CLI."
  exit 1
fi

# Check that local file exists
if [[ ! -f "${LOCAL_FILE}" ]]; then
  echo "Error: local file not found: ${LOCAL_FILE}"
  echo "Run this script from the directory containing the file, or edit LOCAL_FILE."
  exit 1
fi

echo "Checking AWS credentials..."
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "Error: AWS credentials not configured or invalid."
  echo "Run: aws configure"
  exit 1
fi

# Upload the file
aws s3 cp "${LOCAL_FILE}" "${S3_DEST}"

echo "Done. Uploaded to ${S3_DEST}"
