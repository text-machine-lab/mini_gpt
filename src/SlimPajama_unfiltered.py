import os
import argparse
import time
import ast
import json
from collections import Counter

import pandas as pd
from loguru import logger
import datasets
from tqdm.auto import tqdm
import zstandard as zstd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# import text_filter

def save_ds(data, filename, data_path, chunk_id):
    texts = [item['text'] for item in data]
    source = [item['meta']['redpajama_set_name'] for item in data]
    chunk_dataset = datasets.Dataset.from_dict({"text": texts, "src": source})
    save_path = os.path.join(data_path, chunk_id, f"file_{filename}")
    chunk_dataset.save_to_disk(save_path)

def parse_args():
    # python src/main.py
    parser = argparse.ArgumentParser('Filtering  on SlimPajama')

    parser.add_argument('--data_dir',
                    type=str,
                    default='../data/pretraining_data/slimpajama_raw_download/SlimPajama-627B/train',
                    help='Directory for data')

    parser.add_argument('--chunk_id',
                    type=str,
                    default='chunk1',
                    help='which chunk to work with')

    parser.add_argument('--chunk_size',
                    type=int,
                    default=10_000,
                    help='Size of each chunk')

    parser.add_argument(
                    '--data_path',
                    type=str,
                    default='../minigpt/data/pretraining_data/slimpajama_raw_chunks',
                    help='the path where the fetched data is stored')
    return parser.parse_args()

def main():
    args = parse_args()
    folder_path =  os.path.join(args.data_dir, args.chunk_id)

    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        # 1. Decompress the .zst file
        with open(file_path, 'rb') as compressed:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                decompressed_data = reader.read()

            # Write the decompressed data to a temporary file (e.g., temp.txt or temp.json)
            with open("temp.txt", 'wb') as temp_file:
                temp_file.write(decompressed_data)

            data = []
            with open('temp.txt', 'r') as file:
                for line in file:
                    data.append(json.loads(line))

            text_chunk = []

            # logger.info("Iterating over the dataset.")
            for item in tqdm(data):
                text_chunk.append(item)
            save_ds(text_chunk, filename, args.data_path, args.chunk_id)
            if cumulative_tokens >= 2130447357:
                break

if __name__ == "__main__":
    main()