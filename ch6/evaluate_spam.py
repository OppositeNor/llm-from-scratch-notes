import tiktoken
import torch
from config import GPT_CONFIG_124M
from gpt_model import GPTModel, generate
from utils import text_to_token_ids

tokenizer = tiktoken.get_encoding("gpt2")
input_text = "Get your $1000 prize! Click link to claim"

model_path = "models/model_spam.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

model = GPTModel(GPT_CONFIG_124M)
model.out_head = torch.nn.Linear(
    in_features=GPT_CONFIG_124M["emb_dim"],
    out_features=2
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

classifications = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=1,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("This is not a spam" if classifications.flatten()[-1].item() == 0 else "This is a spam")

