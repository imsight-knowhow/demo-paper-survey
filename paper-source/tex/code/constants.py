# --- H100 GPU Specifications ---
BYTES: int = 1  # model weights, we assume FP8
PEAK_FLOPS: float = 1978 * 1e12  # peak flops for FP8
PEAK_FLOPS_ATTENTION: float = 989 * 1e12  # attention is done in BF16
MEMORY_BANDWIDTH: float = 3.35 * (1024**4)  # memory bandwidth in B/s

# --- Transformer Model Configuration ---
HIDDEN_DIM: int = 4096
FFN_DIM: int = 4 * HIDDEN_DIM
NUM_HEADS: int = 32
HEAD_DIM: int = HIDDEN_DIM // NUM_HEADS
NUM_KV_HEADS: int = 8
BYTES_PER_KV_ELEMENT: float = 1  # bytes used per KV element