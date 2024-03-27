import os
import argparse
import time
import ast
import json

import pandas as pd
from loguru import logger
import datasets
from tqdm.auto import tqdm
import zstandard as zstd
from datasets import load_dataset
from tqdm import tqdm

import text_filter

def parse_args():
    # python src/main.py
    parser = argparse.ArgumentParser('Filtering  on SlimPajama')

    parser.add_argument('--data_dir',
                    type=str,
                    default='../data/pretraining_data/slimpajama_raw_download/SlimPajama-627B/train',
                    help='Directory for data')

    parser.add_argument(
        '--max_oov_p',
        type=float,
        default=0.01,
        help='maximum percent of the oov words'
    )
    parser.add_argument(
        '--min_span_len',
        type=int,
        default=32,
        help='minimum length of the filtered sentences'
    )

    parser.add_argument('--chunk_id',
                    type=str,
                    default='chunk1',
                    help='Directory for tokenizer')

    # parser.add_argument('--chunk_size',
    #                 type=int,
    #                 default=10_000,
    #                 help='Size of each chunk')

    parser.add_argument(
        '--filter_data_path',
        type=str,
        default='../data/pretraining_data/slimpajama/filtered/debug_Jan4/',
        help='the path where the filtered data is stored'
    )

    return parser.parse_args()

def filter_texts_batched(items, vocab, min_span_len, max_oov_p, from_sentence_start=True):
    spans = []
    word_count = []
    oov_count = []
    filtered_text = text_filter.filter_vocab(items['text'], vocab, min_span_len, max_oov_p, from_sentence_start)
    for span, info in filtered_text:
        if len(span) > 0:
            spans.append(span)
            word_count.append(info.word_count)
            oov_count.append(info.oov_count)
    # return {"text": "\n".join(spans),
    return {"text": "<s> " + " </s> <s> ".join(spans) + " </s>",
            "word_count": sum(word_count),
            "oov_count": sum(oov_count)}

def process_and_save(texts, vocab, save_path, num_workers, min_span_len, max_oov_p, from_sentence_start):
    # logger.info("Loading texts...")
    chunk_dataset = datasets.Dataset.from_dict({"text": texts})
    # logger.info("Filtering texts...")
    chunk_dataset = chunk_dataset.map(
        filter_texts_batched,
        #batched=True,
        num_proc=num_workers,
        fn_kwargs={"vocab": vocab, "min_span_len": min_span_len, "max_oov_p": max_oov_p, "from_sentence_start": True},
    )

    if chunk_dataset.num_rows != 0:
        chunk_dataset.save_to_disk(save_path)

def main():
    args = parse_args()
    folder_path =  os.path.join(args.data_dir, args.chunk_id)

    # load the vocab
    logger.info("Loading vocab...")
    vocab = pd.read_csv("../data/AOChildes/AOChildes_word_frequency.csv", index_col=0)
    vocab = set(w.lower() for w in vocab["word"] if isinstance(w, str))

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


            c4_chunk = []
            github_chunk = []
            common_crawl_chunk = []
            stack_exchange_chunk = []
            wikipedia_chunk = []
            arxiv_chunk = []
            books_chunk = []
            i =  0


            logger.info("Iterating over the dataset.")
            for item in tqdm(data):
                # logger.info(f"i: {i}")
                if item["meta"]["redpajama_set_name"] == "RedPajamaC4":
                    c4_chunk.append(item["text"])
                if item["meta"]["redpajama_set_name"] == "RedPajamaGithub":
                    github_chunk.append(item["text"])
                if item["meta"]["redpajama_set_name"] == "RedPajamaCommonCrawl":
                    common_crawl_chunk.append(item["text"])
                if item["meta"]["redpajama_set_name"] == "RedPajamaStackExchange":
                    stack_exchange_chunk.append(item["text"])
                if item["meta"]["redpajama_set_name"] == "RedPajamaWikipedia":
                    wikipedia_chunk.append(item["text"])
                if item["meta"]["redpajama_set_name"] == "RedPajamaArXiv":
                    arxiv_chunk.append(item["text"])
                # books dataset
                else:
                    books_chunk.append(item["text"])

            chunk_subsets =  [c4_chunk, github_chunk, common_crawl_chunk, stack_exchange_chunk, wikipedia_chunk, arxiv_chunk, books_chunk]
            for idx, chunk in enumerate(chunk_subsets):
                # save only if the chunk does not exist
                if os.path.exists(os.path.join(args.filter_data_path, args.chunk_id, f"file_{filename}_idx_{idx}")):
                    continue
                process_and_save(
                    chunk,
                    vocab,
                    save_path=os.path.join(args.filter_data_path, args.chunk_id, f"file_{filename}_idx_{idx}"),
                    num_workers=12,
                    min_span_len=args.min_span_len,
                    max_oov_p=args.max_oov_p,
                    from_sentence_start=True
                )

if __name__ == "__main__":
    main()