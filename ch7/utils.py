from matplotlib.ticker import MaxNLocator
import torch
import matplotlib.pyplot as plt

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def plot_losses(epoch_seen, tokens_seen, train_losses, val_losses, save_figure=False):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epoch_seen, train_losses, label="Train loss")
    ax1.plot(epoch_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny() # Create another x axis which shares the y axis
    ax2.plot(tokens_seen, train_losses, alpha=0) # For alignment
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    if save_figure:
        plt.savefig("loss-plot.pdf")
    plt.show()

def plot_values(epoch_seen, examples_seen, train_values, val_values, label="loss", save_figure=False):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epoch_seen, train_values, label=f"Training {label}")
    ax1.plot(epoch_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0) # For alignment
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    if save_figure:
        plt.savefig(f"{label}-plot.pdf")
    plt.show()

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )

        else:
            break
    return correct_predictions / num_examples

def format_input(entry):
    instruction_text = ("Bellow is an instruction that describes a task. Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry["instruction"]}")
    input_text = f"\n\n### Input:\n{entry["input"]}" if entry["input"] else ""
    return instruction_text + input_text

def format_response(entry):
    desired_response = f"\n\n### Response:\n{entry["output"]}"
    return desired_response

def format_entry(entry):
    return format_input(entry), format_response(entry)
