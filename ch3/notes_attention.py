import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66], 
     [0.57, 0.85, 0.64], 
     [0.22, 0.58, 0.33], 
     [0.77, 0.25, 0.10], 
     [0.05, 0.80, 0.55]] 
)

# The attention model
print("Attention model.")

# Calculate queries
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("2nd row attention scores:", attn_scores_2)

# Normalization
attn_weights_2_temp = attn_scores_2 / attn_scores_2.sum()

print("2nd row attention weights:", attn_weights_2_temp)
print("Sum:", attn_weights_2_temp.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("2nd row 1 attention weights:", attn_weights_2)
print("Sum:", attn_weights_2_temp.sum())

# Get the context vector.
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("2nd row context vector:", context_vec_2)

# We can get the attention score of each pair by doing a matrix multiplication:
attn_scores = inputs @ inputs.T
print("Attention scores:")
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print("Attention weights:")
print(attn_weights)
print("All row sums:", attn_weights.sum(dim=-1))

# Get the context vectors from the attention weights
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)


