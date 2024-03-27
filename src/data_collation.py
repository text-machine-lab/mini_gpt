# data collation is based on https://huggingface.co/learn/nlp-course/chapter7/6

import os
import logging
import torch
import transformers
import numpy as np
import random

from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
from functools import partial

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'])

def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # Shifting the inputs and labels to align them happens inside the model,
    # so the data collator just copies the inputs to create the labels.
    # https://discuss.huggingface.co/t/where-does-the-transformers-do-the-target-text-shifting-in-causal-lm/32408/2
    result["labels"] = result["input_ids"].copy()

    return result

def create_data_collator(args, dataset, tokenizer, logger, use_tokenizer):

    # tokenized path for the minigpt slimpajama filtered dataset
    if use_tokenizer == "filtered":
        tokenized_path = '../data/pretraining_data/slimpajama/filtered/debug_Jan4/minigpt_dataset_tokenized/'
    elif use_tokenizer == "unfiltered":
        # tokenized path for the minigpt slimpajama unfiltered dataset
        tokenized_path = '../pretraining_data/pretrain/slimpajama/unfiltered/debug_oct11/minigpt_unfiltered_tokenized'

    print(f"The tokenized path is {tokenized_path}")
    if os.path.exists(tokenized_path):
        lm_dataset = load_from_disk(tokenized_path)
        logger.info("Loading tokenized dataset from disk")
    else:
        column_names = dataset.column_names
        preprocess_function_arg = partial(preprocess_function, tokenizer=tokenizer)
        # for each example in dataset, tokenize the text portion of it
        # features: ['input_ids', 'token_type_ids', 'attention_mask'],
        tokenized_dataset = dataset.map(
            preprocess_function_arg,
            batched=True,
            num_proc=24,
            remove_columns=column_names,
            desc="Tokenizing the dataset",
        )
        group_texts_arg = partial(group_texts, block_size=args.max_seq_len)

        lm_dataset = tokenized_dataset.map(group_texts_arg, batched=True, num_proc=24)
        lm_dataset.save_to_disk(tokenized_path)

    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return data_collator, lm_dataset