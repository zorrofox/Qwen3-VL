"""Static shape constants for the JAX Qwen-VL pipeline."""

# Sequence limits
MAX_SEQ_LEN = 8192
MODEL_MAX_LENGTH = 8192  # default; overridden by training args

# Vision limits
MAX_IMAGES_PER_SAMPLE = 8
MAX_VIDEO_FRAMES = 768

# Token IDs (from Qwen tokenizer)
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652

# Ignore index for label masking
IGNORE_INDEX = -100

# Vision processing defaults
IMAGE_PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MAX_RATIO = 200

IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
