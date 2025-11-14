from torch.nn.attention.flex_attention import BlockMask, create_block_mask

def create_attention_mask_inference(causal_point: int) -> BlockMask:
    # causal_point is the index of the first token in the prediction block  
    def mask_mod(b, h, q_idx, kv_idx):
        is_causal = (kv_idx <= q_idx)
        is_past_causal_point = (causal_point <= q_idx)
        return is_causal | is_past_causal_point
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)