#!/bin/bash

# Usage: ./duplicate_conda_env.sh <new_env_name>
# Example: ./duplicate_conda_env.sh my_env_copy

if [ -z "$1" ]; then
  echo "Usage: $0 <new_env_name>"
  exit 1
fi

NEW_ENV_NAME="$1"
CURRENT_ENV=$(conda info --json | grep -Po '"active_prefix":.*?[^\\]",' | awk -F '"' '{print $4}')

if [ -z "$CURRENT_ENV" ]; then
  echo "Could not detect the current active conda environment."
  exit 2
fi

echo "Exporting environment from: $CURRENT_ENV"
conda list --explicit > env.txt

echo "Creating new environment: $NEW_ENV_NAME"
conda create --name "$NEW_ENV_NAME" --file env.txt

echo "Cleaning up temporary file."
rm env.txt

echo "Done! New environment '$NEW_ENV_NAME' created."