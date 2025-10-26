import json
import os
import urllib.request
from accelerate.utils import tqdm
from datasets import load_dataset
import random

dataset_drop_rate = 0.8

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)
    open_orca_dataset = load_dataset("Open-Orca/OpenOrca", split="train")
    for open_orca_data in tqdm(open_orca_dataset):
        data.append(dict({
            "instruction": open_orca_data["system_prompt"],
            "input": open_orca_data["question"],
            "output": open_orca_data["response"]
        }))
    split_point = int(dataset_drop_rate * len(data))
    print(f"Dropping {split_point}")
    random.shuffle(data)
    data = data[dataset_drop_rate:]
    print("Writing dataset to json file...")
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file)
    return data

file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

data = download_and_load_file(file_path, url)


print("Number of entries:", len(data))

print("Example entry:", data[42])
