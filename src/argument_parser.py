import argparse
from transformers import SchedulerType

def parse_args():
        parser = argparse.ArgumentParser('Train small language models on filtered data')

        parser.add_argument('--tokenizer_path',
                        type=str,
                        default='../tokenizer/SlimPajama/Jan22_15000',
                        help='Directory for tokenizer')

        parser.add_argument('--chkpt_dir',
                        type=str,
                        default='../chkpts/SlimPajama_Nov7/filtered/hidden_512_num_layer_8',
                        help='Directory for checkpoints')

        parser.add_argument('--save_model',
                        action='store_true',
                        default=False,
                        help='Whether or not to save the model')

        parser.add_argument('--restart_frm_chkpt',
                        default=False,
                        action='store_true',
                        help='Resatrt training from checkpoint')

        parser.add_argument('--debug',
                        default=False,
                        action='store_true',
                        help='Debug mode on a smaller dataset')

        parser.add_argument('--debug_sample_size',
                        type=int,
                        default=1000,
                        help='Number of samples to use in debug mode')

        parser.add_argument('--per_device_train_batch_size',
                        type=int,
                        default=16,
                        help='Train Batch size')

        parser.add_argument('--per_device_eval_batch_size',
                type=int,
                default=16,
                help='Eval Batch size')

        parser.add_argument('--total_batch_size',
                type=int,
                default=512,
                help='Batch size')

        parser.add_argument('--max_seq_len',
                        type=int,
                        default=1024,
                        help='max sequence length')

        parser.add_argument('--hidden_size',
                        type=int,
                        default=64,
                        help='Hidden size of the model')


        parser.add_argument('--int_size',
                        type=int,
                        default=256,
                        help='Hidden size of the model')

        parser.add_argument('--rope_theta',
                        type=float,
                        default=10000,
                        help='Hidden size of the model')

        parser.add_argument('--num_layers',
                        type=int,
                        default=2,
                        help='Number of layers in the model')

        parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        help='Number of epochs')

        parser.add_argument('--lr',
                        type=float,
                        default=0.0008,
                        help='Learning rate')

        parser.add_argument("--weight_decay",
                        type=float,
                        default=0.1,
                        help="Weight decay value for AdamW optimizer")

        parser.add_argument("--beta1",
                        type=float,
                        default=0.9,
                        help="Beta1 to use for Adam.")

        parser.add_argument("--beta2",
                        type=float,
                        default=0.95,
                        help="Beta2 to use for Adam.")

        parser.add_argument("--grad_acc_steps",
                        type=int,
                        default=2,
                        help="Accumulate gradient for these many steps")

        parser.add_argument("--eval_every_steps",
                        type=int,
                        default=4000,
                        help="Perform evaluation every n network updates.")

        parser.add_argument("--save_checkpoint_evey_steps",
                        type=int,
                        default=200,
                        help="Save model checkpoint")

        parser.add_argument("--save_total_limit",
                type=int,
                default=10,
                help="Max number of checkpoints to keep, when value is met older checkpoints are deleted")

        parser.add_argument("--logging_steps",
                        type=int,
                        default=10,
                        help="Compute and log training batch metrics every n steps.")

        parser.add_argument("--max_train_steps",
                        type=int,
                        default=None,
                        help="Total number of training steps to perform. If provided, overrides num_epochs.")

        parser.add_argument(
                        "--warmup_percent",
                        type=float,
                        default=0.05,
                        help="Percentage of total steps to warmup LR over.")

        parser.add_argument("--wandb_project",
                        default="small_language_models",
                        help="wandb project name to log metrics to")

        parser.add_argument("--fixed_seed_val",
                           type=int,
                           default=1,
                           help="Value of the seed to fix for reproducibility")

        parser.add_argument("--lr_scheduler_type",
                           type=SchedulerType,
                           default="cosine",
                           help="The scheduler type to use.",
                           choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])

        parser.add_argument("--num_warmup_steps",
                           type=int,
                           default=0,
                           help="Number of steps for the warmup in the lr scheduler.")


        parser.add_argument("--use_tokenizer",
                        type=str,
                        default="filtered",
                        help="whether to use filtered or unfiltered tokenizer")

        return parser.parse_args()