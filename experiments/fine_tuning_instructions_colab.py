from functools import partial
import datetime
import json
import os
import time
import tiktoken
import torch
import shutil
from torch.utils.data import DataLoader

from dataset import InstructionDataset, custom_collate
from train_utils import calc_loss_loader, train_model_autocast
from config import use_config, model_size
from gpt_model import GPTModel, generate
from utils import format_input, plot_losses, text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if device == torch.device("cpu"):
    num_cpus = os.cpu_count()
    if num_cpus is None:
        print("Failed to get CPU core count.")
    else:
        torch.set_num_threads(num_cpus-1)
        print(f"{num_cpus} CPU cores found, using {num_cpus-1} cores.")

print("Using device:", device)
tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(42)
load_checkpoint = True
checkpoint_path = f"checkpoints/checkpoint_instruction_{model_size}.pth"
model_output_path = f"models/model_instruction_{model_size}.pth"
pretrained_model_path = f"models/model_pretrain_{model_size}.pth"
dataset_path = "instruction-data.json"
num_epochs = 6

num_workers = 0
batch_size = 8

if load_checkpoint and os.path.exists(checkpoint_path):
    model = GPTModel(use_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = torch.GradScaler()
    scaler.load_state_dict(checkpoint["scaler"])
    print("Checkpoint loaded:", checkpoint_path)
    del checkpoint
else:
    model = GPTModel(use_config).to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    scaler = torch.GradScaler()

def prepare_dataset():
    with open(dataset_path, "r") as f:
        data = json.load(f)

    train_portion = int(len(data) * 0.80)
    test_portion = int(len(data) * 0.15)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    val_data = data[train_portion:train_portion+val_portion]
    test_data = data[train_portion+val_portion:]
    del data

    print("Training set size:", len(train_data))
    print("Validation set size:", len(val_data))
    print("Test set size:", len(test_data))

    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)
    test_dataset = InstructionDataset(test_data, tokenizer)
    return train_dataset, val_dataset, test_dataset, val_data

def prepare_dataloaders():
    print(f"Loading dataset {dataset_path}...")
    train_dataset, val_dataset, test_dataset, val_data = prepare_dataset()
    custom_collate_fn = partial(custom_collate, device=device, allowed_max_length=use_config["context_length"])

    print("Preparing dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, val_data

train_loader, val_loader, test_loader, val_data = prepare_dataloaders()

input_text = format_input(val_data[0])
print(input_text)
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=35,
    context_size=use_config["context_length"],
    eos_id=50256
)
print(token_ids_to_text(token_ids, tokenizer)[len(input_text):])

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

print("Train loss:", train_loss)
print("Validation loss:", val_loss)
print("Test loss:", test_loss)

start_time = time.time()

model.train()

train_losses, val_losses, tokens_seen = train_model_autocast(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs,
    eval_freq=50,
    eval_iter=5,
    start_context=format_input(val_data[0]),
    tokenizer=tokenizer,
    scaler=scaler,
)

end_time = time.time()
duration_minutes = (end_time - start_time) / 60
print(f"Training completed in {duration_minutes:.2f} minutes.")

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

print("Train loss:", train_loss)
print("Validation loss:", val_loss)
print("Test loss:", test_loss)

torch.save(model.state_dict(), model_output_path)

print("Model saved:", model_output_path)

torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    'scaler': scaler.state_dict(),
}, checkpoint_path)

print("Checkpoint saved:", checkpoint_path)

plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, save_figure=True)

shutil.copy(checkpoint_path, "/content/drive/MyDrive/")
shutil.copy("loss-plot.pdf", f"/content/drive/MyDrive/loss-plot-{datetime.datetime.now()}-{model_size}.pdf")
print("Checkpoint saved to google drive.")
