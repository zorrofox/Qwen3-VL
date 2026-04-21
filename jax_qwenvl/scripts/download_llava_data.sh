#!/bin/bash
# Download LLaVA-Instruct-150K dataset (annotations + COCO images).
#
# Sets up directory structure:
#   /data/llava/
#     llava_instruct_150k.json    (annotations)
#     train2017/                  (COCO images)
#       000000000009.jpg
#       ...
#
# Usage:
#   bash jax_qwenvl/scripts/download_llava_data.sh [DATA_ROOT]
#
# DATA_ROOT defaults to /data/llava

set -euo pipefail

DATA_ROOT="${1:-/data/llava}"
echo "=== LLaVA-Instruct-150K Dataset Download ==="
echo "Data root: ${DATA_ROOT}"

mkdir -p "${DATA_ROOT}"

# 1. Download LLaVA-Instruct-150K annotations from HuggingFace
ANNOTATION_FILE="${DATA_ROOT}/llava_instruct_150k.json"
if [ -f "${ANNOTATION_FILE}" ]; then
    echo "Annotations already exist: ${ANNOTATION_FILE}"
else
    echo "Downloading LLaVA-Instruct-150K annotations..."
    pip install -q huggingface_hub 2>/dev/null || true
    python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='liuhaotian/LLaVA-Instruct-150k',
    filename='llava_instruct_150k.json',
    repo_type='dataset',
    local_dir='${DATA_ROOT}',
)
print(f'Downloaded annotations to: {path}')
"
    echo "Annotations downloaded: ${ANNOTATION_FILE}"
fi

# Verify annotation format
echo "Verifying annotation format..."
python3 -c "
import json
with open('${ANNOTATION_FILE}') as f:
    data = json.load(f)
print(f'Total samples: {len(data)}')
sample = data[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Image: {sample.get(\"image\", \"N/A\")}')
if 'conversations' in sample:
    print(f'Conversations: {len(sample[\"conversations\"])} turns')
    print(f'  First turn from: {sample[\"conversations\"][0][\"from\"]}')
"

# 2. Download COCO train2017 images
COCO_DIR="${DATA_ROOT}/train2017"
if [ -d "${COCO_DIR}" ] && [ "$(ls -1 ${COCO_DIR}/*.jpg 2>/dev/null | head -1)" ]; then
    echo "COCO images already exist: ${COCO_DIR}"
    echo "  Image count: $(ls -1 ${COCO_DIR}/*.jpg | wc -l)"
else
    echo "Downloading COCO train2017 images (~18GB)..."
    cd "${DATA_ROOT}"

    COCO_ZIP="train2017.zip"
    if [ ! -f "${COCO_ZIP}" ]; then
        wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip -O "${COCO_ZIP}"
    fi

    echo "Extracting COCO train2017..."
    unzip -q -o "${COCO_ZIP}"
    rm -f "${COCO_ZIP}"

    echo "COCO images ready: $(ls -1 ${COCO_DIR}/*.jpg | wc -l) images"
fi

# 3. Verify the image path format matches annotations
echo ""
echo "Verifying image paths..."
python3 -c "
import json, os
with open('${ANNOTATION_FILE}') as f:
    data = json.load(f)
# Check first 5 samples
missing = 0
checked = min(20, len(data))
for i in range(checked):
    img = data[i].get('image', '')
    # LLaVA annotations use format like '000000215677.jpg'
    # or 'coco/train2017/000000215677.jpg'
    full_path = os.path.join('${DATA_ROOT}', img)
    if not os.path.exists(full_path):
        # Try with train2017/ prefix
        alt_path = os.path.join('${DATA_ROOT}', 'train2017', img)
        if os.path.exists(alt_path):
            print(f'Image paths need train2017/ prefix (sample: {img})')
            break
        else:
            missing += 1
            if missing <= 3:
                print(f'Missing: {img} (tried {full_path} and {alt_path})')
if missing == 0:
    print('All checked images found!')
else:
    print(f'{missing}/{checked} images missing')
"

echo ""
echo "=== Download Complete ==="
echo "Annotation: ${ANNOTATION_FILE}"
echo "Images:     ${COCO_DIR}"
echo ""
echo "To register this dataset, add to jax_qwenvl/data/__init__.py:"
echo "  LLAVA_INSTRUCT_150K = {"
echo "      \"annotation_path\": \"${ANNOTATION_FILE}\","
echo "      \"data_path\": \"${DATA_ROOT}\","
echo "  }"
