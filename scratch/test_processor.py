import sys
import os
import numpy as np
from PIL import Image

# Add jax_qwenvl to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoProcessor
from jax_qwenvl.data.data_processor import update_processor_pixels

def main():
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"Loading processor for {model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Failed to load processor from HF: {e}")
        print("Trying to create a dummy processor or use local path...")
        # If HF fails, we might need a local path or skip.
        # Let's try to find if there is a local cache or path.
        return

    # Create a dummy image
    img = Image.fromarray(np.uint8(np.random.randint(0, 255, (1000, 1000, 3))))
    img_path = "/tmp/test_img.png"
    img.save(img_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Test 1: Default pixels
    print("\n--- Test 1: Default pixels ---")
    res1 = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="np")
    if "pixel_values" in res1:
        print(f"Default pixel_values shape: {res1['pixel_values'].shape}")
    else:
        print("No pixel_values in output")

    # Test 2: Updated pixels to 50176
    print("\n--- Test 2: Updated pixels to 50176 ---")
    class DummyDataArgs:
        min_pixels = 28 * 28 * 16
        max_pixels = 50176
        video_min_pixels = 256 * 28 * 28
        video_max_pixels = 1664 * 28 * 28

    data_args = DummyDataArgs()
    processor = update_processor_pixels(processor, data_args)

    res2 = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="np")
    if "pixel_values" in res2:
        print(f"Updated pixel_values shape: {res2['pixel_values'].shape}")
    else:
        print("No pixel_values in output")

    # Clean up
    if os.path.exists(img_path):
        os.remove(img_path)

if __name__ == "__main__":
    main()
