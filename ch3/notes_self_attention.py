import torch
from self_attention import CausalAttention, MultiHeadAttentionWrapper, SelfAttentionV2

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66], 
     [0.57, 0.85, 0.64], 
     [0.22, 0.58, 0.33], 
     [0.77, 0.25, 0.10], 
     [0.05, 0.80, 0.55]] 
)
# Self attention model
print("Self attention model")

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2   # The output embedding, usually is the same as the input embedding, but using 2 here.

# Define the W_q, W_k and W_v matrices and initialize them to random (disabling gradient for demonstrations)
torch.manual_seed(42)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("2nd row query:", query_2)

# Calcualte the keys and values of the input
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# The attention score is the dot product of query and key.
keys_2 = keys[1]
attn_score_2_2 = query_2.dot(keys_2)
print("2nd-2nd attenetion score:", attn_score_2_2)

# We can get all the attention scores by matrix multiplication.
attn_scores_2 = query_2 @ keys.T
print("2nd row attention score:", attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / (d_k**0.5), dim=-1)
print(attn_weights_2)

# Calculate the context vector
context_vec_2 = attn_weights_2 @ values
print("2nd row context vector:", context_vec_2)

# Use the self-defined self attention model.
torch.manual_seed(42)
sa_v2 = SelfAttentionV2(d_in, d_out)
print("Self-defined model:")
print(sa_v2(inputs))

# First approach: apply softmax -> mask the upper diagonal with 0s -> normalize
# Get attention scores.
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print("Attention weights:")
print(attn_weights)

# Apply a causal attention mask, mask the above diagonal with 0s.
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("mask_simple:")
print(mask_simple)
masked_simple = mask_simple * attn_weights
print("masked_simple:")
print(masked_simple)

# Then normalize the rows
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("masked_simple_norm:")
print(masked_simple_norm)

"""
Masked score in the softmax calculation -> Information leakage?
No. Because the distribution of data are the same after applying the softmax normalization,
and another normalization was done after, so no information leakage.
"""

# Second apprach: Mask with -inf above diagonal -> apply softmax. (exp(-inf) = 0)
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("masked:")
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights) # Value the same as masked_simple_norm
# This appraoch only has to do one normalization.

# Drop out 50%
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print("dropout(example):")
print(dropout(example)) # Note that every non-dropped out element is upscaled by 1/0.5
print("dropout(attn_weights):")
print(dropout(attn_weights))

# Causal attention class
# Make sure the class can handle more than one inputs (stack two inputs together to test)
batch = torch.stack((inputs, inputs), dim=0)
print("batch.shape")
print(batch.shape)

torch.manual_seed(42)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)

# Multi-head attention
torch.manual_seed(42)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
