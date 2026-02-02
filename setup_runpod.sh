#!/bin/bash

echo "Starting RunPod Environment Setup..."

# 1. Update and install basic tools
apt-get update && apt-get install -y git wget curl

# 2. Check GPU Status
if command -v nvidia-smi &> /dev/null
then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: GPU not detected. Training might not work."
fi

# 3. Upgrade pip and install dependencies
echo "Installing Python dependencies (this may take a few minutes)..."
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Create necessary directories
mkdir -p data models evaluation_results scripts

# 5. Verify data files
if [ -f "data/train_triplets.jsonl" ]; then
    echo "Success: Data files found."
else
    echo "WARNING: data/train_triplets.jsonl not found. Please upload your 'data' folder."
fi

echo "--------------------------------------"
echo "Environment setup complete!"
echo "You can now run: python scripts/train_peft.py"
echo "--------------------------------------"
