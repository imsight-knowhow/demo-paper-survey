def gemm_roofline(m: int, n: int, k: int) -> tuple[float, float]:
    "Returns the runtime (in s) and flops of doing a `C=A@B` with A/B of shape `[m, k]/[k, n]` "
    total_IO = (
        m * k * BYTES + # read A
        n * k * BYTES + # read B
        m * n * BYTES   # write C
    )
    total_flops = 2 * m * n * k
    return max(total_IO / MEMORY_BANDWIDTH, total_flops / PEAK_FLOPS), total_flops

# Attention roofline analysis
def calculate_fused_attention_roofline(block_size: int, kv_length: int, batch_size: int) -> float:
    num_tokens = block_size * batch_size
    # Memory in Q: (batch_size, block_size,  NUM_HEADS, HEAD_DIM)
    bytes_Q = num_tokens * NUM_HEADS * HEAD_DIM * BYTES
    # Memory in K: (batch_size, kv_length, NUM_KV_HEADS, HEAD_DIM)
    bytes_K = batch_size * kv_length * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_KV_ELEMENT
    # Flops P=QK^T
    _, flops_QKt = gemm_roofline(num_tokens, kv_length, NUM_HEADS * HEAD_DIM)    
    # Softmax memory transfer is ignored as it's not read or stored in memory, flops negliglbe
    bytes_softmax = 0
    # Memory in V: (batch_size, kv_length, NUM_KV_HEADS, HEAD_DIM)
    bytes_V = batch_size * kv_length * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_KV_ELEMENT
    # Flops PV
    _, flops_PV = gemm_roofline(num_tokens, kv_length, NUM_HEADS * HEAD_DIM)    
    # Memory out PV: (batch_size, block_size, NUM_HEADS, HEAD_DIM)
    bytes_PV = num_tokens * NUM_HEADS * HEAD_DIM * BYTES

    total_memory = bytes_Q + bytes_K + bytes_V + bytes_PV + bytes_softmax
    total_flops = flops_QKt + flops_PV
    return max(total_memory / MEMORY_BANDWIDTH, total_flops / PEAK_FLOPS_ATTENTION)


# FFN roofline analysis
def calculate_linear_layers_roofline(block_size: int, batch_size: int) -> float:
    num_tokens = block_size * batch_size
    # FFN: Up-projection
    total_time = gemm_roofline(num_tokens, HIDDEN_DIM, HIDDEN_DIM * FFN_DIM)[0]
    # FFN: Down-projection
    total_time += gemm_roofline(num_tokens, HIDDEN_DIM * FFN_DIM, HIDDEN_DIM)[0]
    # Attention linear layers
    total_time += gemm_roofline(num_tokens, HIDDEN_DIM, NUM_HEADS * HEAD_DIM)[0] # Q-proj
    total_time += gemm_roofline(num_tokens, HIDDEN_DIM, NUM_HEADS * NUM_KV_HEADS)[0] # K-proj
    total_time += gemm_roofline(num_tokens, HIDDEN_DIM, NUM_HEADS * NUM_KV_HEADS)[0] # V-proj
    total_time += gemm_roofline(num_tokens, NUM_HEADS * HEAD_DIM, HIDDEN_DIM)[0] # out-proj
    return total_time