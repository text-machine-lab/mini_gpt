import argparse
import logging
import os
import sys
import time
import random
import numpy as np
import torch
import transformers
import wandb
import math
from tqdm import tqdm

import datasets
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, TrainingArguments, SchedulerType, AdamW, get_scheduler, get_cosine_schedule_with_warmup
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from optimum.bettertransformer import BetterTransformer
from accelerate import Accelerator
from accelerate.utils import set_seed


from logging_config import configure_logger
from argument_parser import parse_args
from data_collation import create_data_collator
from utils import init_weights



def initialize(args, logger):
    transformers.set_seed(args.fixed_seed_val)

def main():
    # Initialize
    args = parse_args()
    # Set up logging and log arguments
    logger = configure_logger(__name__)
    logger.info('Starting up...')
    logger.info('Arguments: {}'.format(args))
    initialize(args, logger)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    # intialize accelerator
    # The code for using accelerator is from
    # https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
    accelerator = Accelerator(
        log_with = "wandb",
        project_dir = args.chkpt_dir,
        mixed_precision = 'bf16',
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.fixed_seed_val is not None:
        set_seed(args.fixed_seed_val)

    if accelerator.is_main_process and args.chkpt_dir is not None:
        os.makedirs(args.chkpt_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # load the minigpt dataset
    if args.use_tokenizer == "filtered":
        args.dataset_path = '../data/pretraining_data/slimpajama/filtered/debug_Jan4/minigpt_dataset/'
    elif args.use_tokenizer == "unfiltered":
        args.dataset_path = '../pretraining_data/pretrain/slimpajama/unfiltered/debug_oct11/'

    dataset = load_from_disk(args.dataset_path)
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)


    # tokenization of the data
    with accelerator.main_process_first():
        data_collator, lm_dataset = create_data_collator(args, dataset, tokenizer, logger, args.use_tokenizer)
    logger.info("Created data collator and dataset")
    train_test_lm_dataset = lm_dataset.train_test_split(test_size=0.003, seed=args.fixed_seed_val)
    print(f"The  length of the dataset is {len(train_test_lm_dataset)}")
    # create dataloader with data_collator as collate_fn to create batches
    train_dataset = DataLoader(train_test_lm_dataset["train"], batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)
    eval_dataset = DataLoader(train_test_lm_dataset["test"], batch_size=args.per_device_train_batch_size, collate_fn=data_collator)
    print(f"The length of the train dataset before accelerate is {len(train_dataset)}")

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        intermediate_size=args.int_size,
        rope_theta = args.rope_theta,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=8,
    )

    model = LlamaForCausalLM(config=config)
    model = BetterTransformer.transform(model)
    num_params = sum(param.numel() for param in model.parameters())


    print(f"Created model with config: {config}")
    num_params = sum(param.numel() for param in model.parameters())
    print(f"Number of parameters: {num_params}")

    optimizer = AdamW(params=model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps = 10e-5,
                    weight_decay=args.weight_decay)

    gradient_accumulation_steps = args.total_batch_size / (args.per_device_train_batch_size * accelerator.num_processes)
    print(f"The number of gradient accumulation steps is {gradient_accumulation_steps}")

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_dataset) / gradient_accumulation_steps
    print(f"Number of update steps per epoch is {num_update_steps_per_epoch}")
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps,
    num_cycles=0.25
    )

    model, optimizer, train_dataset, eval_dataset, lr_scheduler = accelerator.prepare(
                   model, optimizer, train_dataset, eval_dataset, lr_scheduler)


    if not gradient_accumulation_steps.is_integer():
        raise ValueError("The calculated number of gradient accumulation steps should be a an integer ")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    experiment_config["gradient_accumulation_steps"] = gradient_accumulation_steps
    accelerator.init_trackers("small_language_models", experiment_config)

    logger.info("***** Running training *****")
    logger.info("Num epochs = %s", args.num_epochs)
    logger.info("Batch size = %s", args.per_device_train_batch_size)
    logger.info("Total Batch size = %s", args.total_batch_size)
    logger.info("Total optimization steps = %s", args.max_train_steps)


    global_step = 0
    tokens_seen = 0
    pad_idx = tokenizer.encode(tokenizer.pad_token)[0]
    for epoch in tqdm(range(args.num_epochs)):
        total_loss = 0
        for step, batch in enumerate(train_dataset):
            logger.info(f"batch in step {step}")
            tokens_seen += ((batch["input_ids"] != pad_idx).sum().item() * accelerator.num_processes)
            # gradient accumulation is already done for you by the accelerator
            with accelerator.accumulate(model):
                model.train()
                if 'token_type_ids' in batch: # for llama config only, do this to avoid error
                    del batch['token_type_ids']
                outputs = model(**batch)
                # accelerator takes care of loss co-ordination during training
                loss = outputs.loss
                logger.info(f"Loss: {outputs.loss.item()}")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1

            accelerator.log({"train_loss": loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "tokens_seen": tokens_seen},
                step=global_step,
            )

            # evaluate every eval_every steps and at the last step
            if global_step % args.eval_every_steps == 0 or global_step >= args.max_train_steps and global_step != 0:
                losses = []
                model.eval()
                with torch.no_grad():
                    for eval_batch in eval_dataset:
                        if 'token_type_ids' in eval_batch: # for llama config only, do this to avoid error
                            del eval_batch['token_type_ids']
                        eval_outputs = model(**eval_batch)

                        batch_loss = eval_outputs.loss

                        losses.append(accelerator.gather_for_metrics(batch_loss.repeat(args.per_device_eval_batch_size)))

                    losses = torch.cat(losses)
                    eval_loss = torch.mean(losses)
                    try:
                        perplexity = math.exp(eval_loss)
                    except OverflowError:
                        perplexity = float("inf")
                    logger.info(f"Eval loss: {eval_loss}")
                    logger.info(f"Perplexity: {perplexity}")
                    accelerator .log({"eval_loss": eval_loss,
                            "perplexity": perplexity},
                            step=global_step)

                    if args.save_model:
                        # save model checkpoint only when not performing hyperparameter search
                        saved_model_path = os.path.join(args.chkpt_dir, f"step{global_step}")
                        logger.info("Saving model checkpoint to %s", saved_model_path)
                        if not os.path.exists(saved_model_path):
                            unwrapped_model = accelerator.unwrap_model(model)
                            # reverse model that is wrapped by BetterTransformer
                            reversed_model = BetterTransformer.reverse(unwrapped_model)
                            reversed_model.save_pretrained(saved_model_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
                            print("Saved model")
            if global_step >= args.max_train_steps:
                break
            ############### end of training loop #####################

    saved_model_path = os.path.join(args.chkpt_dir, f"final_model")
    # unwrap model wrapped by accelerate
    unwrapped_model = accelerator.unwrap_model(model)
    # reverse model that is wrapped by BetterTransformer
    reversed_model = BetterTransformer.reverse(unwrapped_model)
    reversed_model.save_pretrained(saved_model_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    logger.info("Training complete")
    accelerator.end_training()

if __name__ == "__main__":
    main()