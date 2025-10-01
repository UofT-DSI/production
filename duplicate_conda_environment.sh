
#!/bin/bash

# Usage: ./duplicate_conda_environment.sh <new_env_name> [env_file]
# Example: ./duplicate_conda_environment.sh my_env_copy my_env_spec.txt

if [ -z "$1" ]; then
  echo "Usage: $0 <new_env_name> [env_file]"
  exit 1
fi

NEW_ENV_NAME="$1"
ENV_FILE="${2:-conda_env_spec.txt}"

# Get the current active environment name
CURRENT_ENV=$(conda info --json | grep -Po '"active_prefix":.*?[^\\]",' | awk -F '"' '{print $4}')

if [ -z "$CURRENT_ENV" ]; then
  echo "Could not detect the current active conda environment."
  exit 2
fi

echo "Exporting environment from: $CURRENT_ENV to $ENV_FILE"
conda list --explicit > "$ENV_FILE"
echo "Environment spec saved to $ENV_FILE. You can commit this file and use it in other branches or projects."

echo "To create a new environment elsewhere, run:"
echo "  conda create --name <new_env_name> --file $ENV_FILE"

echo "Done!"