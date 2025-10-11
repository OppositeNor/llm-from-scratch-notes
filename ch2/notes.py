import urllib.request
import re
from tokenizer import SimpleTokenizerV1
from importlib.metadata import version
import tiktoken

# Retrieve "The Verdict" text
url = ("https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# Open the file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

# Self defined tokenizer

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Create a vocabulary for the input text.
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_tokens)
vocab = {token:integer for integer,token in enumerate(all_tokens)}

tokenizer = SimpleTokenizerV1(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

# BPE tokenizer

print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
print(len(enc_text))

enc_sample = enc_text[50:]

# Print context desired pair
context_size = 4
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "-->", tokenizer.decode([desired]))

