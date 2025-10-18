import torch
import tiktoken
from utils import plot_losses, text_to_token_ids, token_ids_to_text
from train_utils import calc_loss_loader, train_model_simple
from config import GPT_CONFIG_124M
from gpt_model import GPTModel, generate, generate_text_simple
from dataset import create_dataloader_v1
import matplotlib.pyplot as plt

torch.manual_seed(42)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

# Utility functions

start_context = "every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor(
    [[16833, 3626, 6100], # every effort moves
     [40, 1107, 588]])    # I really like
targets = torch.tensor(
    [[3626, 6100, 345 ],  # effort moves you
     [1107, 588, 11311]]) # really like chocolate

# Evaluate the model, and get the probability distribution of the output token.
with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token ids:\n")
print(token_ids)
print(f"Target batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Output batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# Print the softmax probability corresponding to the target score:
target_probas_1 = probas[0, [0, 1, 2], targets[0]]
print("Text 1:", target_probas_1)

target_probas_2 = probas[1, [0, 1, 2], targets[1]]
print("Text 2:", target_probas_2)

# We get the loss by performing the following steps:
# Logits -- softmax -->
# Probabilities ->
# Target probabilities -- log -->
# Log probabilities -- average -->
# Average log probabilities -- negative -->
# Negative average log probabilities
# The lesser the negative average log probability is, the larget the probability.

# Log probabilities:
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
# Average log probabilities:
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
# Negative average log probabilities:
neg_avg_log_probas = -1 * avg_log_probas
print(neg_avg_log_probas)

# This is also called the cross entropy loss, which is already provided by pytorch:
print("Logits shape:", logits.shape)
print("Target shape:", targets.shape)
# For the cross_entropy loss in pytorch, we first flatten the tensors
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits shape:", logits_flat.shape)
print("Flattened targets shape:", targets_flat.shape)
# The cross_entropy loss will take the softmax by itself, so we can directly pass the original tensors:
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("Loss:", loss)

# "Perplexity" is generally used to evaluate the stability of the model, which measures the expectation number
# of tries by the model to get the target value (by drawing samples from the distribution). The definition is:
# exp(L) where L is the corss entropy loss. In the book it's kind of vaguely explained, so the following is my
# understanding (and some help from ChatGPT).

# The cross entropy loss is defined as:
# $$L = - \frac{1}{n} \sum_{i=1}^n \log (p_i)$$
# where taking exp(L) is:
# $$e^{- \frac{1}{n} \sum_{i=1}^n \log (p_i)} = (\prod_{i=1}^n p_i)^{-\frac{1}{n}}$$
# Which is the inverse of the geometric average of all posibilities, which is the expecting number of tries
# the model takes to output the target.

# In the book it says that perplexity measures how uncertain the model is for the predicted result, saying
# "being unsure about which among" how many tokens in the vocabulary "to generate as the next token".
# However, I don't think that's really accurate, because if we have the following situation:
# [0.1, 0.8, 0.1] target = 0
# [0.8, 0.1, 0.1] target = 2
# [0.0, 0.1, 0.8] target = 1
# The model is pretty certain that the output should be [1, 0, 2], but the perpexity is:
# $$(0.1 * 0.1 * 0.1)^{-\frac{1}{3}}$$
# which is 10, and we don't even have 10 choices (we only have 3). So I believe "expecting number of tries
# to get the target output" is more rigorous, which also accord to the mathematical definition of "The inverse
# of probability".

# Another thing to notice is that in practice we generally don't use the formula:
# $$(\prod_{i=1}^n p_i)^{-\frac{1}{n}}$$
# to calculate perplexity, since the product of all probabilities might be very small. Usually we still use the
# exp(L) (where L is the cross entropy loss) to calculate perplexity.

perplexity = torch.exp(loss)
print("Perplexity:", perplexity)
# Which outputs 60974.2305, which we only have 50257 vocabularies, so "it is struggling on 60974.2305 target
# outputs" doesn't make sense.

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.9
split_idx = int(train_ratio * total_characters)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(42)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation layer:")
for x, y in val_loader:
    print(x.shape, y.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# Train the model
if True:
    torch.manual_seed(42)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) # The AdamW optimizer
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # Plot the curves for the losses
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # Text generation
    # We can bring the model to CPU when evaluating (since it's not too heavy)
    model.to("cpu")
    model.eval()
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:")
    print(token_ids_to_text(token_ids, tokenizer))

# Probabilistic sampling
# Example:
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

# Greedy encoding:
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[int(next_token_id)])

# Get from probability:
torch.manual_seed(42)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[int(next_token_id)])

def print_sampled_tokens(probas):
    torch.manual_seed(42)
    sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)

# Temparature scaling
def softmax_with_temperature(logits, temparature):
    scaled_logits = logits / temparature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f"Temperature = {T}")
ax.set_ylabel("Probability")
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
# plt.show()
# Smaller temperature will have more certaincy. Larger temperature will be more evenly distributed.

# Top k sampling
# Replaces the other logits other than the top k ones to -inf, and softmax will evaluate them to 0.
# In other words, only sample from the top k probabilities.
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top k logits:", top_logits)
print("Top k positions:", top_pos)
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],   # For all the logits less than the minimum of the top k tokens
    input=torch.tensor(float("-inf")),              # Replace them with with -inf
    other=next_token_logits                         # For all other logits, retain them from next_token_logits
)
print(new_logits)
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

torch.manual_seed(42)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print(token_ids_to_text(token_ids, tokenizer))
