"""
Build and train the BPE tokenizer
based on - https://huggingface.co/docs/tokenizers/quicktour and
https://text-machine-lab.github.io/nlp_class_2022/schedule/ (Machine Translation)
"""
# python create_tokenizer.py --vocab_size 32_000 --save_dir ./tokenizer/Nov_21/
import argparse
import logging
import transformers
import datasets
from datasets import load_dataset, load_from_disk

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer

#setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()

def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Train a tokenizer")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=f'../data/pretraining_data/slimpajama/filtered/debug_Jan4/minigpt_dataset/'
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Size of the vocabulary"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory which will be used to save tokenizer."
    )

    args = parser.parse_args()

    return args


def main():
    """Build and train tokenizer
    """

    args = parse_args()
    logger.info("Start training tokenizer with args %s", args)
    logger.info("Loading Minigpt dataset")
    dataset = load_from_disk(args.dataset_path)

    # tokenize and train
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer_trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[MASK]", "<s>", "</s>"],
                                    vocab_size=args.vocab_size)
    tokenizer.pre_tokenizer = tokenizer.pre_tokenizer = Whitespace()

    iterator = (item['text'] for item in dataset)

    tokenizer.train_from_iterator(iterator, trainer = tokenizer_trainer)

    # wrap the tokenizer to make it usable in HuggingFace Transformers
    tokenizer = transformers.PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, mask_token="[MASK]",
                bos_token="<s>", eos_token="</s>", pad_token="[PAD]")

    # save tokenizer
    logger.info("Saving tokenizer in %s", args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

if __name__ == "__main__":
    main()