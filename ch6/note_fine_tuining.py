import time
import tiktoken
import torch
from torch.utils.data import DataLoader
from train_utils import calc_loss_loader_last, train_classifier_simple
from utils import calc_accuracy_loader, plot_values, text_to_token_ids, token_ids_to_text
from config import GPT_CONFIG_124M
from gpt_model import GPTModel, generate_text_simple
from dataset import SpamDataset

pretrained_model_path = "models/model_pretrain.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
)
print(train_dataset.max_length)

val_dataset = SpamDataset(
    csv_file="val.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
)

test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
)

num_workers = 0
torch.manual_seed(42)
batch_size = 8

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False
)

print(f"{len(train_loader)} training batches.")
print(f"{len(val_loader)} validation batches.")
print(f"{len(test_loader)} test batches.")

# Load the pretrained model.
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
model.eval()

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))

# Freeze the model
for param in model.parameters():
    param.requires_grad = False

# Alternate the final last layer of the model. Instead of outputing text, output
# a 0 (not spam) or 1 (spam)
torch.manual_seed(42)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=GPT_CONFIG_124M["emb_dim"],
    out_features=2
)

# In the book, the author claimed, enabeling the last layer of transformer block and the final
# normalization layer will increase performance. So:
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs shape:", inputs.shape)

with torch.no_grad():
    outputs = model(inputs)
print("Outputs:")
print(outputs)
print("Outputs dimension:", outputs.shape)
print("Last output token:", outputs[:, -1, :])
probas = torch.softmax(outputs[:, -1, :], dim=-1)
print("Probabilities:", probas)
label = torch.argmax(probas)
print("Class label:", label.item())

model.to(device)

torch.manual_seed(42)
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
print(f"Train accuracy: {train_accuracy}")
print(f"Val accuracy: {val_accuracy}")
print(f"Test accuracy: {test_accuracy}")

with torch.no_grad():
    train_loss = calc_loss_loader_last(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader_last(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader_last(test_loader, model, device, num_batches=5)
print("Train loss:", train_loss)
print("Validation loss:", val_loss)
print("Test loss:", test_loss)

start_time = time.time()
torch.manual_seed(42)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen =\
    train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq=50, eval_iter=5)
end_time = time.time()
train_duration_minutes = (end_time - start_time) / 60
print(f"Training completed in {train_duration_minutes:.2f} minutes.")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss", save_figure=True)
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy", save_figure=True)

train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
print(f"Train accuracy: {train_accuracy}")
print(f"Val accuracy: {val_accuracy}")
print(f"Test accuracy: {test_accuracy}")
