#!/bin/bash
# Download Qwen3-VL model from HuggingFace Hub.
#
# Sets up directory structure:
#   ~/models/Qwen3-VL-2B-Instruct/
#     config.json
#     model-00001-of-00002.safetensors
#     model-00002-of-00002.safetensors
#     model.safetensors.index.json
#     tokenizer.json
#     ...
#
# Usage:
#   bash jax_qwenvl/scripts/download_model.sh [MODEL_NAME] [MODEL_DIR]
#
# MODEL_NAME defaults to Qwen/Qwen3-VL-2B-Instruct
# MODEL_DIR  defaults to ~/models/<model_short_name>

set -euo pipefail

MODEL_NAME="${1:-Qwen/Qwen3-VL-2B-Instruct}"
MODEL_SHORT="${MODEL_NAME##*/}"  # e.g. Qwen3-VL-2B-Instruct
MODEL_DIR="${2:-${HOME}/models/${MODEL_SHORT}}"

echo "=== Qwen3-VL Model Download ==="
echo "Model:  ${MODEL_NAME}"
echo "Target: ${MODEL_DIR}"

mkdir -p "${MODEL_DIR}"

# Check if model is already downloaded
if [ -f "${MODEL_DIR}/config.json" ] && ls "${MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
    echo "Model already exists: ${MODEL_DIR}"
    echo "  config.json: $(ls -lh "${MODEL_DIR}/config.json" | awk '{print $5}')"
    SAFETENSOR_COUNT=$(ls -1 "${MODEL_DIR}"/*.safetensors 2>/dev/null | wc -l)
    echo "  safetensors: ${SAFETENSOR_COUNT} files"
    echo ""
    echo "To re-download, remove the directory first:"
    echo "  rm -rf ${MODEL_DIR}"
    exit 0
fi

# Download using huggingface_hub snapshot_download
echo "Downloading ${MODEL_NAME} ..."
python3 -c "
from huggingface_hub import snapshot_download
import os, shutil

# Download to HF cache first
cache_dir = snapshot_download('${MODEL_NAME}')
print(f'Downloaded to cache: {cache_dir}')

# Copy/symlink to target directory
target = '${MODEL_DIR}'
for fname in os.listdir(cache_dir):
    src = os.path.join(cache_dir, fname)
    dst = os.path.join(target, fname)
    if os.path.isfile(src):
        # Resolve symlinks (HF cache uses symlinks)
        real_src = os.path.realpath(src)
        if not os.path.exists(dst):
            shutil.copy2(real_src, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f'  Copied: {fname} ({size_mb:.1f} MB)')
        else:
            print(f'  Exists: {fname}')

print('Done.')
"

# Verify download
echo ""
echo "Verifying download..."
python3 -c "
import os, json

model_dir = '${MODEL_DIR}'

# Check config.json
config_path = os.path.join(model_dir, 'config.json')
if not os.path.isfile(config_path):
    print('ERROR: config.json not found!')
    exit(1)

with open(config_path) as f:
    config = json.load(f)

model_type = config.get('model_type', 'unknown')
hidden_size = config.get('hidden_size', config.get('text_config', {}).get('hidden_size', '?'))
num_layers = config.get('num_hidden_layers', config.get('text_config', {}).get('num_hidden_layers', '?'))
print(f'Model type:    {model_type}')
print(f'Hidden size:   {hidden_size}')
print(f'Num layers:    {num_layers}')

# Check safetensors
safetensors = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
total_size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in safetensors)
print(f'Safetensors:   {len(safetensors)} files ({total_size / (1024**3):.2f} GB)')

# Check tokenizer
has_tokenizer = os.path.isfile(os.path.join(model_dir, 'tokenizer.json'))
print(f'Tokenizer:     {\"found\" if has_tokenizer else \"MISSING\"}')"

echo ""
echo "=== Download Complete ==="
echo "Model path: ${MODEL_DIR}"
echo ""
echo "Use this path for training:"
echo "  MODEL_PATH=${MODEL_DIR} bash jax_qwenvl/scripts/train_tpu.sh"
