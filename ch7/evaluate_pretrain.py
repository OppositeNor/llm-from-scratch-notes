import tiktoken
import torch
from config import use_config
from gpt_model import GPTModel, generate
from utils import text_to_token_ids, token_ids_to_text

tokenizer = tiktoken.get_encoding("gpt2")
input_text = "Every effort moves you"

model_path = "models/model_pretrain.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

model = GPTModel(use_config)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=15,
    context_size=use_config["context_length"],
    top_k=25,
    temperature=1.4
)
print(token_ids_to_text(token_ids, tokenizer))
