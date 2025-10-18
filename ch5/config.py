GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,      # We shorten to 256 tokens instead of 1024
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,             # The number of transformer blocks.
    "drop_rate": 0.1,
    "qkv_bias": False
}
