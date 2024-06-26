# Emergent Abilities in Reduced-Scale Generative Language Models


This repository has code to filter data from exisiting corpora based on child vocabulary and train small language models on this filtered data.

<p align="center">
    <img
         src="images/method.png"
         width="100%"
         height="100%">
</p>

## Installation

```bash
git clone git@github.com:text-machine-lab/mini_gpt.git
cd mini_gpt
pip install -r requirements.txt
```

## Usage

The tokenizer for filtering is from [filter_vocab_cpp](https://github.com/Guitaricet/filter_vocab_cpp).<br>
Compile the  C++ based filtration code and copy over the object fileto the src directory.

The vocabulary used for simplification of pre-training data  can be found in ```data/AOChildes_word_frequency.csv``` This vocabulary is based on child-directed speech transcripts that can be found [here](https://github.com/UIUCLearningLanguageLab/AOCHILDES)

## Downloading the SlimPajama dataset using git lfs:
```bash
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
```
Chunks 1-10 are downloaded when this command is run.

For gathering the unfiltered dataset:
```bash
python SlimPajama_unfiltered.py
```

## For Vocab filtering, use the following command per chunk:
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

## Pre-training data:
The pre-training data which consits of vocabulary filtered SlimPajama dataset can be found here [22B](https://huggingface.co/datasets/text-machine-lab/vocab_filtered_dataset_22B) and [2.1B](https://huggingface.co/datasets/text-machine-lab/vocab_filtered_dataset_2.1B)

## To train BPE tokenizer:
```bash
python create_tokenizer.py --dataset_path ./dataset \
    --vocab_size 15_000 \
    --save_dir ./tokenizer
```

## For pre-training a language model with distributed training:
```bash
python -u -m accelerate.commands.launch main.py \
     --lr 2.8e-3 --num_warmup_steps 1000 --num_layers 8 \
     --hidden_size 32 --use_tokenizer filtered \
     --chkpt_dir ../models/SlimPajama_Nov23_context128_vocab_21k/filtered/hidden_32_num_layer_8_int_128 \
     --int_size 128 --rope_theta 20
```
## Notebooks
For creating the minigpt dataset, counting the number of tokens in the filtered dataset, analysis of the filtered dataset use the notebook ```notebooks/2.0-dataset-statistics.ipynb```

For applying position interpolation on pre-trained models use the notebook
```notebooks/1.0-rope-pi.ipynb```

To filter downstream  evaluation datasets based on AO-Childes vocabulary use the notebook
```notebooks/4.0-dataset_filtering.ipynb```

To get generations from the pre-trained and baseline models use
```notebooks/3.0-model-generations.ipynb```

## Citation
```
@misc{muckatira2024emergent,
      title={Emergent Abilities in Reduced-Scale Generative Language Models},
      author={Sherin Muckatira and Vijeta Deshpande and Vladislav Lialin and Anna Rumshisky},
      year={2024},
      eprint={2404.02204},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```