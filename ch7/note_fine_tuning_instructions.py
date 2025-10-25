from functools import partial
import json
import os
import time
import tiktoken
import torch
from torch.utils.data import DataLoader
from train_utils import calc_loss_loader, train_model_simple
from gpt_model import GPTModel, generate
from config import use_config
from dataset import InstructionDataset, custom_collate
from utils import format_entry, format_input, plot_losses, text_to_token_ids, token_ids_to_text

with open("instruction-data.json", "r") as f:
    data = json.load(f)

model_input, desired_output = format_entry(data[50])
print(model_input + desired_output)

model_input, desired_output = format_entry(data[999])
print(model_input + desired_output)

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion+test_portion]
val_data = data[train_portion+test_portion:]

print("Training set size:", len(train_data))
print("Test set size:", len(test_data))
print("Validation set size:", len(val_data))

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

def custom_collate_draft_v1(batch, device, pad_token_id=50256):
    batch_max_length = max(len(item) for item in batch)
    inputs_list = []
    for item in batch:
        padded = item + [pad_token_id] * (batch_max_length - len(item))
        inputs = torch.tensor(padded)
        inputs_list.append(inputs)
    inputs_tensor = torch.stack(inputs_list).to(device)
    return inputs_tensor

print(custom_collate_draft_v1(batch, device="cpu"))

def custom_collate_draft_v2(batch, device, pad_token_id=50256):
    batch_max_length = max(len(item) for item in batch)
    inputs_list, targets_list = [], []
    for item in batch:
        padded = item + [pad_token_id] * (batch_max_length - len(item))
        inputs = torch.tensor(padded)
        targets = torch.tensor(padded[1:] + [pad_token_id])
        inputs_list.append(inputs)
        targets_list.append(targets)
    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)
    return inputs_tensor, targets_tensor

inputs_tensor, targets_tensor = custom_collate_draft_v2(batch, device="cpu")
print(inputs_tensor)
print(targets_tensor)

inputs_tensor, targets_tensor = custom_collate(batch, device=torch.device("cpu"))
print(inputs_tensor)
print(targets_tensor)

targets_1 = torch.tensor([0, 1])
targets_2 = torch.tensor([0, 1, 1])
targets_3 = torch.tensor([0, 1, -100])
logits_1 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5]]
)
logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]
)
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
# -100 is ignored by cross_entropy by default, so loss_1 and loss_3 should be the same
print("loss_1, loss_2, loss_3:", loss_1, loss_2, loss_3)
# It used to be common masking out the instruction during loss calculations; however,
# the paper "Instruction Tuning With Loss Over Instructions"(https://arxiv.org/abs/2405.14394)
# showed that not masking out instructions will have better performance. So we are not
# masking it here.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
if device == torch.device("cpu"):
    cpu_count = os.cpu_count()
    if cpu_count is None:
        print("Cannot get CPU count")
    else:
        torch.set_num_threads(cpu_count - 1)
        print(f"CPU count: {cpu_count}")
print("Using device:", device)

custom_collate_fn = partial(custom_collate, device=device, allowed_max_length=use_config["context_length"])

num_workers = 0
batch_size = 8
torch.manual_seed(42)

train_dataset = InstructionDataset(train_data, tokenizer)
val_dataset = InstructionDataset(val_data, tokenizer)
test_dataset = InstructionDataset(test_data, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=True,
    # In the book this is set to True; however, since we have custom_collate which pads the
    # input and target, dropping the last one seems unecessary. My understanding is that,
    # if we have a batch that is very small in size, the cross entropy will have large variance,
    # which is not ideal for training.
    drop_last=True,
    num_workers=num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=False,
    # But we can set it to False for validation and testing.
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

torch.manual_seed(42)
model = GPTModel(use_config)
model.load_state_dict(torch.load("models/model_pretrain.pth", map_location=device))
model.eval()

input_text = format_input(val_data[0])
print(input_text)
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=use_config["context_length"],
    eos_id=50256
)
print(token_ids_to_text(token_ids, tokenizer)[len(input_text):])

model.to(device)
model.train()
torch.manual_seed(42)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

print("Train loss:", train_loss)
print("Validation loss:", val_loss)
print("Test loss:", test_loss)

start_time = time.time()
torch.manual_seed(42)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 30

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[1]), tokenizer=tokenizer
)

end_time = time.time()
duration_minutes = (end_time - start_time) / 60
print(f"Training completed in {duration_minutes:.2f} minutes.")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
