from loguru import logger
import torch
import math
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaConfig,
    BitsAndBytesConfig,
)
from peft import (
    PromptTuningConfig,
    PrefixTuningConfig,
    LoraConfig,
    IA3Config,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
)

# custom classses
from src.modeling_llama import LlamaForCausalLM
from src.pos_emb_classes import (
    ScaledLlamaRotaryEmbedding,
    NTKAwareRope,
    NTKAwareByParts,
    XPOS,
)

#
class CustomLlamaConfig(LlamaConfig):
    def __init__(self, _flash_attn_2_enabled, **kwargs):
        super().__init__(**kwargs)
        self._flash_attn_2_enabled = _flash_attn_2_enabled

def get_model(
        model_name_or_path: str="",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        load_in_8bit=False,
        device_map=None,
        train=False,
        peft_method="lora",
        lora_r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0,
        lora_bias="none",
        _flash_attn_2_enabled=True,
        config=None,
):

    # check free space
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB-2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    # quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
    )

    # define model
    logger.info(f"Starting to load the model ({model_name_or_path})...")
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    if _flash_attn_2_enabled:
        logger.info("Using Flash-Attention.")
        config = CustomLlamaConfig(_flash_attn_2_enabled=_flash_attn_2_enabled, **config.__dict__)

    # @TODO: instead of loading the model from LlamaForCausalLM class we need to, define config and then load model based on config
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        #quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    logger.info(f"Loaded the model ({model_name_or_path}).")
    dtypes = {}
    for name, param in model.named_parameters():
        dtype = param.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += param.numel()
        param.requires_grad = False

    # keep track of parameter data types
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        logger.info(f"{k} {v} {v / total}")

    # if we want to fine-tune the model then we only fine-tune the LoRa (or othe peft) parameters
    if train:
        # define peft config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
        )

        # patch peft with model
        model = get_peft_model(model, peft_config)
        logger.info(f"Created PEFT modules (LoRA) in the model.")

    # verifying the size of the model
    par_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    par_fixed = sum(p.numel() for p in model.parameters())
    par_percent = int(100 * par_trainable / par_fixed)
    logger.info(f"Total number of trainable parameters: {par_trainable:,} ({par_percent}%)")

    # log model
    logger.info(model)

    return model

def get_position_emb(
    pos_emb_name: str,
    seq_len_test: int,
    dim: int,
    base: int,
    scale_power: int,
    device: torch.device,
    seq_len_train: int = None,
    scale = None,
):
    if scale is None:
        scale = math.ceil(max(1, seq_len_test / seq_len_train))

    #
    if pos_emb_name == "alibi":
        NotImplementedError("ALiBi style position embeddings are not implemented yet")
        return

    elif pos_emb_name == "ntk_aware":
        pos_emb = NTKAwareRope(
            dim=dim,
            scale=scale,
            scale_power=scale_power,
            max_position_embeddings=seq_len_test,
            base=base,
            device=device,
        )

    elif pos_emb_name == "ntk_aware_by_parts":
        pos_emb = NTKAwareByParts(
            dim=dim,
            scale=scale,
            scale_power=scale_power,
            max_position_embeddings=seq_len_test,
            base=base,
            device=device,
        )

    elif pos_emb_name == "xpos":
        pos_emb = LlamaXPosAttention(
            scale_base=scale_base,
        )

    else:
        # default option
        pos_emb = ScaledLlamaRotaryEmbedding(
            dim=dim,
            scale=scale,
            scale_power=scale_power,
            max_position_embeddings=seq_len_test,
            base=base,
            device=device,
        )

    return pos_emb

def patch_model_with_rope(
    pos_emb_name: str,
    model: nn.Module,
    seq_len_train: int,
    seq_len_test: int,
    scale_power: int,
    base: int = 10000,
):
    # calculate scale
    scale = math.ceil(max(1, seq_len_test / seq_len_train))

    # Patch the custom pos_em with the model
    if pos_emb_name == "alibi":
        NotImplementedError("ALiBi style position embeddings are not implemented yet")
        return

    elif pos_emb_name == "ntk_aware":
        pos_emb = NTKAwareRope.patch(
            model=model,
            scale=scale,
            max_position_embeddings=seq_len_test,
        )

    elif pos_emb_name == "ntk_aware_by_parts":
        pos_emb = NTKAwareByParts.patch(
            model=model,
            seq_len_train=seq_len_train,
            scale=scale,
            max_position_embeddings=seq_len_test,
        )


    elif pos_emb_name == "xpos":
        pos_emb = LlamaXPosAttention.patch(
            model=model,
            scale_base=scale_base,
        )

    else:
        # default option
        pos_emb = ScaledLlamaRotaryEmbedding.patch(
            model=model,
            scale=scale,
            scale_power=scale_power,
        )

    return model