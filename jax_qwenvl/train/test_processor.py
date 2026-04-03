import sys
import os
import numpy as np
from PIL import Image

# Add jax_qwenvl to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from transformers import AutoProcessor
from jax_qwenvl.data.data_processor import update_processor_pixels

def main():
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"Loading processor for {model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Failed to load processor: {e}")
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
    res1 = processor.apply_chat_template(messages, tokenize=True, return_tensors="np")
    if "pixel_values" in res1:
        print(f"Default pixel_values shape: {res1['pixel_values'].shape}")
    else:
        print("No pixel_values in output")

    # Test 2: passing processor_kwargs to apply_chat_template
    print("\n--- Test 2: passing processor_kwargs to apply_chat_template ---")
    try:
        res2 = processor.apply_chat_template(
            messages, 
            tokenize=True, 
            return_tensors="np", 
            processor_kwargs={"max_pixels": 50176}
        )
        if "pixel_values" in res2:
            print(f"ApplyChatTemplate with processor_kwargs pixel_values shape: {res2['pixel_values'].shape}")
        else:
            print("No pixel_values in output")
    except Exception as e:
        print(f"Test 2 failed: {e}")

    # Test 3: Two-step process with processor_kwargs in __call__
    print("\n--- Test 3: Two-step process with processor_kwargs in __call__ ---")
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="np",
            processor_kwargs={"max_pixels": 50176}
        )
        if "pixel_values" in inputs:
            print(f"Two-step with processor_kwargs pixel_values shape: {inputs['pixel_values'].shape}")
        else:
            print("No pixel_values in output")
    except Exception as e:
        print(f"Test 3 failed: {e}")

    # Test 4: passing max_pixels inside the message content dict
    print("\n--- Test 4: passing max_pixels inside the message content dict ---")
    try:
        messages_with_max_pixels = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path, "max_pixels": 50176},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        res4 = processor.apply_chat_template(
            messages_with_max_pixels, 
            tokenize=True, 
            return_tensors="np"
        )
        if "pixel_values" in res4:
            print(f"ApplyChatTemplate with max_pixels in message pixel_values shape: {res4['pixel_values'].shape}")
        else:
            print("No pixel_values in output")
    except Exception as e:
        print(f"Test 4 failed: {e}")

    # Test 5: Two-step process with max_pixels in message content
    print("\n--- Test 5: Two-step process with max_pixels in message content ---")
    try:
        from qwen_vl_utils import process_vision_info
        messages_with_max_pixels = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path, "max_pixels": 50176},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        image_inputs, video_inputs = process_vision_info(messages_with_max_pixels)
        
        text = processor.apply_chat_template(
            messages_with_max_pixels, tokenize=False, add_generation_prompt=False
        )
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="np",
        )
        if "pixel_values" in inputs:
            print(f"Two-step with max_pixels in message pixel_values shape: {inputs['pixel_values'].shape}")
        else:
            print("No pixel_values in output")
    except Exception as e:
        print(f"Test 5 failed: {e}")

    # Test 6: Set processor.image_processor.max_pixels directly and use apply_chat_template with PIL Image
    print("\n--- Test 6: Set processor.image_processor.max_pixels directly and use apply_chat_template with PIL Image ---")
    try:
        pil_img = Image.open(img_path)
        
        # Reset processor to clean state if needed, or just update it
        processor.image_processor.max_pixels = 50176
        
        messages_pil = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        
        res6 = processor.apply_chat_template(
            messages_pil, 
            tokenize=True, 
            return_tensors="np"
        )
        if "pixel_values" in res6:
            print(f"Test 6 pixel_values shape: {res6['pixel_values'].shape}")
        else:
            print("No pixel_values in output")
            
        # Check if it was resized
        print(f"Processor max_pixels is set to: {processor.image_processor.max_pixels}")
    except Exception as e:
        print(f"Test 6 failed: {e}")

    if os.path.exists(img_path):
        os.remove(img_path)

if __name__ == "__main__":
    main()
