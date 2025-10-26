import os
import torch
import tiktoken
from utils import plot_losses
from config import use_config, model_size
from dataset import create_dataloader_v1
from gpt_model import GPTModel
from train_utils import train_model_simple

model_output_path = f"models/model_pretrain_{model_size}.pth"
checkpoint_path = f"checkpoints/checkpoint_pretrain_{model_size}.pth"
load_checkpoint = True

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

tokenizer = tiktoken.get_encoding("gpt2")

train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=use_config["context_length"],
    stride=use_config["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=use_config["context_length"],
    stride=use_config["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

torch.manual_seed(42)

if load_checkpoint and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GPTModel(use_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1) # The AdamW optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
else:
    model = GPTModel(use_config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1) # The AdamW optimizer

model.train()

num_epochs = 30
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Plot the curves for the losses
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

torch.save(model.state_dict(), model_output_path)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}, checkpoint_path)
