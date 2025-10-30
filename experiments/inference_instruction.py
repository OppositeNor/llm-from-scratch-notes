import tiktoken
import torch
from config import use_config, model_size
from gpt_model import GPTModel, generate
from utils import format_entry, text_to_token_ids, token_ids_to_text

tokenizer = tiktoken.get_encoding("gpt2")

model_path = f"models/model_instruction_{model_size}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
max_token = 1024

torch.manual_seed(42)

model = GPTModel(use_config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

instruction_text = input("Instruction: ")
input_text = input("Input: ")

input, response = format_entry({
    "instruction": instruction_text,
    "input": input_text,
    "output": ""
})
entry = input + response

result = entry

for _ in range(max_token):
    with torch.no_grad():
        token = token_ids_to_text(generate(
            model=model,
            idx=text_to_token_ids(result, tokenizer).to(device),
            max_new_tokens=1,
            context_size=use_config["context_length"]
        ), tokenizer)
        if "<|endoftext|>" in token:
            break
        result = token

print(result[len(entry):])
