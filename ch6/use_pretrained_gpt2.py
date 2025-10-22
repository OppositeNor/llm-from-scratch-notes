import tiktoken
import torch
from gpt_download import download_and_load_gpt2
from train_utils import load_weights_to_gpt
from config import GPT_CONFIG_124M
from gpt_model import GPTModel, generate
from utils import text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
model_output_path = "models/model_pretrain.pth"

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)" # Load the gpt2 small model
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()
load_weights_to_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(42)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:", token_ids_to_text(token_ids, tokenizer))


torch.save(gpt.state_dict(), model_output_path)

