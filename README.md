# mini_gpt

This repository has code to filter data from exisiting corpora based on child vocabulary and train small language models on this filtered data

## Installation

```bash
git clone git@github.com:SherinBojappa/small_language_models.git
cd small_language_models
pip install -r requirements.txt
```

## Usage

The tokenizer for filtering is from [filter_vocab_cpp](https://github.com/Guitaricet/filter_vocab_cpp).<br>
The object file for the text tokenizer is in ```src/text_filter.cpython-311-x86_64-linux-gnu.so```<br>

Downloading the SlimPajama dataset using git lfs:
```bash
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
```
Chunks 1-10 are downloaded when this command is run.

For gathering the unfiltered dataset:
```bash
python SlimPajama_unfiltered.py
```

For Vocab filtering:
<!-- ```bash
python src/vocab_utils.py --dataset_name redpajama \
     --subset common_crawl \
     --filter_data_path minigpt
``` -->
```bash
python SlimPajama_filtering.py  --chunk_id 1
```

<!-- Updated ipthon notebook can be found here - work in progress:
```bash
notebooks/filtering_dev_aug14.ipynb
``` -->

For creating the minigpt dataset, counting the number of tokens in the filtered dataset, analysis of the filtered dataset and the tokenizers use the notebook ```notebooks/count_tokens.ipynb```

To train BPE tokenizer:
```bash
python create_tokenizer.py --dataset_path ./dataset \
    --vocab_size 15_000 \
    --save_dir ./tokenizer
```

For training a language model with distributed training:
```bash
python -u -m accelerate.commands.launch main.py \
     --lr 2.8e-3 --num_warmup_steps 1000 --num_layers 8 \
     --hidden_size 32 --use_tokenizer filtered \
     --chkpt_dir ../models/SlimPajama_Nov23_context128_vocab_21k/filtered/hidden_32_num_layer_8_int_128 \
     --int_size 128 --rope_theta 20
```