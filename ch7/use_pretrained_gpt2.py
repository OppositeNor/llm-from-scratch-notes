import tiktoken
import torch
from gpt_download import download_and_load_gpt2
from train_utils import load_weights_to_gpt
from config import use_config, model_size
from gpt_model import GPTModel, generate
from utils import text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
model_output_path = f"models/model_pretrain_{model_size}.pth"

settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)

gpt = GPTModel(use_config)
gpt.eval()
load_weights_to_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(42)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=use_config["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:", token_ids_to_text(token_ids, tokenizer))


torch.save(gpt.state_dict(), model_output_path)

