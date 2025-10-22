import os
import time
import tiktoken
import torch
from torch.utils.data import DataLoader

from config import GPT_CONFIG_124M
from dataset import SpamDataset
from gpt_model import GPTModel
from train_utils import train_classifier_simple
from utils import plot_values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model_path = "models/model_pretrain.pth"
model_output_path = "models/model_spam.pth"
checkpoint_path = "checkpoints/checkpoint_spam.pth"
load_checkpoint = True

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


print("Using device:", device)

torch.manual_seed(42)


# Alternate the final last layer of the model. Instead of outputing text, output
# a 0 (not spam) or 1 (spam)
torch.manual_seed(42)
num_classes = 2

if load_checkpoint and os.path.exists(checkpoint_path):
    print(f"Continuing from checkpoint: {checkpoint_path}")
    model = GPTModel(GPT_CONFIG_124M)
    model.out_head = torch.nn.Linear(
        in_features=GPT_CONFIG_124M["emb_dim"],
        out_features=2
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1) # The AdamW optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
else:
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    model.out_head = torch.nn.Linear(
        in_features=GPT_CONFIG_124M["emb_dim"],
        out_features=2
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1) # The AdamW optimizer

# Freeze the model
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last two layers of transformer blocks and the final output layer
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.out_head.parameters():
    param.requires_grad = True

train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
)

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

start_time = time.time()
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen =\
    train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq=50, eval_iter=5)
end_time = time.time()
train_duration_minutes = (end_time - start_time) / 60
print(f"Training completed in {train_duration_minutes:.2f} minutes.")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss", save_figure=True)

torch.save(model.state_dict(), model_output_path)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}, checkpoint_path)
