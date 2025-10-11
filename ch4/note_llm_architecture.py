import tiktoken
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gpt_model import GELU, FeedForward, GPTModel, LayerNorm, TransformerBlock
from config import GPT_CONFIG_124M

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print("batch:")
print(batch)

# layer normalization example
torch.manual_seed(42)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
print(batch_example)
out = layer(batch_example)
print(out)

# Get mean and variance
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# Use the self-implemented LayerNorm
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# GELU activation function
gelu, relu = GELU(), nn.ReLU()

# Create 100 sample data from -3 to 3
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
# plt.show()

ffn = FeedForward(GPT_CONFIG_124M)
# Create a sample input with batch dimension 2
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

# Shortcut
# Adding an x to the result output of a layer could prevent gradient loss
# y = f(x) + x -> dy / dx = df/dx(x) + 1
# Which we see that there's always a "+1", which even if the gradient is
# lossed after inputing to the layer, we still have a "+1" to get the
# gradient of the input parameters.

# Example of a network that has a shortcut:
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[5], layer_sizes[6]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[6], layer_sizes[7]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[7], layer_sizes[8]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[8], layer_sizes[9]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                # Perform shortcut
                x = x + layer_output
            else:
                x = layer_output
        return x

# Helper functiton to print out gradient.
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.0]])

    # Calculate loss based on MSE loss
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backprop the loss to get gradient.
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

layer_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[-1.0, 0.0, 1.0]])

torch.manual_seed(42)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

print("Gradient of model_without_shortcut")
# Notice that some of the gradients becomes very small
print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(42)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

print("Gradient of model_with_shortcut")
# Doesn't have "very small gradients" problems
print_gradients(model_with_shortcut, sample_input)

# Transformer block
torch.manual_seed(42)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)

# The final GPT model.
torch.manual_seed(42)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}") # 163,009,536

# The reason it is not 124M is because the GPT2 model uses a "weight tying"
# mechanism, which reuses the weight in the embedding layer in the output
# layer. So if we subtract the output layer's weight:
total_params_2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_2}") # 124,412,160
# Which becomes the number of weights we expected (124M). In the book, the
# author claimed that using separate token embedding and output layer usually
# results in a better performance. So weight embedding is not used here.

total_size_bytes = total_params * 4 # Assuming 4 bytes per weight.
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

# Text generation
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] # Focuses only on the last token.

        # Since softmax is monotonic, this step applying softmax is redundent. Here it just showing
        # converting logits into a PDF.
        probas = torch.softmax(logits, dim=-1)
        # Pick the most likely token, which is also called greedy decoding
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1) # Append the new generated token to the running sequence
    return idx

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add a dimension for the batch.
print("Encoded tensor shape:", encoded_tensor.shape)

model.eval() # Put the model into evaluation mode (disables features like dropout)
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
