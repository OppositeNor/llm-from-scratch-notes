import urllib.request
import torch
from dataset import create_dataloader_v1

# Retrieve "The Verdict" text
url = ("https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# Open the file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))

# Set up some random seed
torch.manual_seed(1)

# Vocabulary size
vocab_size = 50257

# Output embedding vector size
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInput shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("Token embedding shape:", token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Position embedding shape:", pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("Input embedding shape:", input_embeddings.shape)

