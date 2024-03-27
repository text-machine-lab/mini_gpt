from loguru import logger

from numpy import pi as PI
import math

import src.modeling_llama
from src.modeling_llama import rotate_half
import torch
import torch.nn as nn


def truncate_frequency(f, t, low, z):
    ft = torch.where(f < t, low, f)
    ft = torch.where(f < t / z, 0, ft)
    return ft

def batch_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    position_ids = position_ids.squeeze(0)
    cos = cos[:, :, position_ids, :]  # [bs, 1, seq_len, dim]
    sin = sin[:, :, position_ids, :]  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def fixed_pos_embedding(seq_len, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, dtype=torch.float32) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float32), inv_freq)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)



class ScaledLlamaRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        scale: float = 1.0, scale_power: float = 0.0, truncate: int = 0, randomize: bool = False, max_position_embeddings: int = 4096, base=10000, device=None, freq_2pi=None,
    ):
        super().__init__()
        self.scale = scale
        self.scale_power = scale_power
        self.randomize = randomize

        # calculate angle
        # TODO: following line does not have '2' in the power
        """
        theta = 10000 ^ ( -2 * (i / d) ),
        where,
        i ~ position (but as the complex numbers are formed by pairing two dimesions, i only takes d/2 unique values)
        d ~ dimension of the space where q and k are defined (this is = hidden_dmi // num_heads)
        10000 = base
        """
        if freq_2pi is None:
            freq_2pi = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))


        # adjust the angle based on the scale
        """
        This operation is the Position Interpolation (PI) method (equation 4 in https://arxiv.org/pdf/2306.15595.pdf).

        In PI, the RoPE function is defined as follows,
        f(x, m), where 'x' is the vector we want to rotate and 'm' is the position of the vector in a sequence,

        RoPE with PI is defined as follows,
        f'(x, m) = f(x, m * (L_test / L_train)) = f(x, m * scale), basically squishing the position of the vector 'x'

        Because, the rotation angle is multiplied by the position, we can either scale the theta or the position 'm',
        effect will be the same.

        Here, we scale the theta.

        """
        if self.scale > 1.0:
            freq_2pi /= self.scale

        # adjust the angle based on power scale
        """

        """

        #
        if self.scale_power > 0:
            scale_power_tensor = (1.0 - torch.arange(len(freq_2pi)).to(device) / len(freq_2pi)) ** scale_power
            freq_2pi *= scale_power_tensor
        if (truncate or 0) > 0:
            cutoff = 2 * PI / truncate
            freq_2pi = truncate_frequency(freq_2pi, cutoff, 2 * PI / (truncate * 16), 8)
        self.register_buffer("freq_2pi", freq_2pi)
        # Build here to make `torch.jit.trace` work.
        self.cache_buffers(max_position_embeddings)
        self.rebuild_random = True
        self.inv_freq = freq_2pi

    def cache_buffers(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.freq_2pi.device, dtype=self.freq_2pi.dtype)
        pos_x_freq = torch.einsum("i,j->ij", t, self.freq_2pi)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((pos_x_freq, pos_x_freq), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def random_init_forward(self, x):
        if not self.rebuild_random:
            return

        t = torch.cumsum(torch.rand(
            (x.shape[0], x.shape[2]), device=self.freq_2pi.device, dtype=self.freq_2pi.dtype) * 2, -1)
        limit = torch.tensor(x.shape[2])
        t *= limit / torch.maximum(t[:, -1:], limit)
        pos_x_freq = torch.einsum("bi,j->bij", t, self.freq_2pi)
        emb = torch.cat((pos_x_freq, pos_x_freq), dim=-1)
        self.register_buffer('cos_random', emb.cos().to(dtype=x.dtype)[:, None, ...])
        self.register_buffer('sin_random', emb.sin().to(dtype=x.dtype)[:, None, ...])
        self.rebuild_random = False

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.randomize and self.training:
            self.random_init_forward(x)
            return self.cos_random, self.sin_random

        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.cache_buffers(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype))

    @staticmethod
    def patch(model: torch.nn.Module, scale: float = 1.0, scale_power: float = 0.0, truncate: int = 0, randomize: bool = False, config=None):
        if config is None:
            config = model.config
        model = getattr(model, 'base_model', model)
        dim = config.hidden_size / config.num_attention_heads

        #
        logger.info("Postion Interpolation - Rotary Position Embedding hyperparameters")
        logger.info(f"Dimension: {dim}")
        logger.info(f"Scale (for the position): {scale}")
        logger.info(f"Scale Power: {scale_power}")
        logger.info(f"Max Pos Emb: {config.max_position_embeddings}")
        rotary_emb = ScaledLlamaRotaryEmbedding(
            dim,
            scale=scale,
            scale_power=scale_power,
            truncate=truncate,
            randomize=randomize,
            max_position_embeddings=config.max_position_embeddings,
            device=model.device,
        )
        for decoder in model.layers:
            assert hasattr(decoder, 'self_attn') and hasattr(decoder.self_attn, 'rotary_emb')
            assert decoder.self_attn.rotary_emb.inv_freq.shape == rotary_emb.freq_2pi.shape
            decoder.self_attn.rotary_emb = rotary_emb
        if randomize:
            try:
                from fastchat.train import llama_flash_attn_monkey_patch
                llama_flash_attn_monkey_patch.apply_rotary_pos_emb = batch_apply_rotary_pos_emb
            except ImportError:
                pass
            modeling_llama.saved_apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
            modeling_llama.apply_rotary_pos_emb = batch_apply_rotary_pos_emb

            def randomize_hook(module: torch.nn.Module, _: tuple):
                # Need kwargs to do this without the lazy caching with rebuild_random
                if module.training:
                    rotary_emb.rebuild_random = True
            rotary_emb.random_init_hook = model.register_forward_pre_hook(randomize_hook)
        return rotary_emb


class XPOS(nn.Module):
    def __init__(self, head_dim, max_position_embeddings, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.max_seq_len_cached = max_position_embeddings
        self.cached = False

    def set_scale_base(self, new_base):
        self.scale_base = new_base
        self.cached = False

    def cache_buffers(self, length: int):
        self.register_buffer(
            "scale", (torch.arange(0, self.head_dim, 2, dtype=torch.float32) + 0.4 * self.head_dim) / (1.4 * self.head_dim)
        )
        # The reason for not doing a simple [0, length) is because of float16 limitations.
        min_pos = -length // 2
        max_pos = length + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(length, self.head_dim // 2)
        self.register_buffer('scale_cached', scale, persistent=False)
        self.register_buffer('inv_scale_cached', 1.0 / scale, persistent=False)
        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)
        self.cached = True

    def forward(self, x: torch.Tensor, offset, downscale=False):
        with torch.autocast('cuda', enabled=False), torch.device(x.device):
            length = offset + x.shape[2]
            if not self.cached:
                self.cache_buffers(self.max_seq_len_cached)
            # It is unsafe to grow the buffers after allocation due to the offset issue.
            if length > self.max_seq_len_cached:
                raise NotImplementedError('Cannot increase buffer after initialization.')
                # self.cache_buffers(length)

            scale = self.inv_scale_cached if downscale else self.scale_cached
            cos = self.cos_cached
            sin = self.sin_cached
            if scale.shape[0] > length:
                scale = scale[offset:length]
                sin = sin[offset:length]
                cos = cos[offset:length]

            return apply_rotary_pos_emb(x, sin, cos, scale)


class NTKAwareRope(ScaledLlamaRotaryEmbedding):
    def __init__(
        self,
        dim: int,
        scale: float = 1.0, scale_power: float = 0.0, truncate: int = 0, randomize: bool = False, max_position_embeddings: int = 4096, base: int=10000, device=None,
    ):

        """
        In RoPE, complex numbers are defined with pairs such as (d_0, d_1), (d_2, d_3) ...
        i.e. (d_0 + i * d_1), (d_2 + i * d_3) ... likewise.

        This NLTK-aware RoPE modifies the base of the complex numbers.
        i.e. base = ((seq_len_test / seq_len_train) ** (dim / dim-2)) * base

        Paragraph from YaRN paper:
        As we want the lowest frequency to be scaled as much as linear positional scaling and the highest frequency to stay constant,
        we need to find a new base b' such that the last dimension matches the wavelength of linear interpolation with a scale factor s.
        Since the original RoPE method skips odd dimensions in order to concatenate both cos(2πx/λ) and sin(2πx/λ) components into a single embedding,
        the last dimension d ∈ D is |D| - 2
        """
        new_base = base * (scale ** (dim / (dim-2)))

        #
        super().__init__(
            base=new_base,
            max_position_embeddings=max_position_embeddings,
            dim=dim,
            scale=1, # 1 because we already accounted the effect of scale in new_base
            scale_power=scale_power,
            device=device,
        )

        return

    @staticmethod
    def patch(
        model: torch.nn.Module,
        scale: float = 1.0,
        scale_power: float = 0.0,
        truncate: int = 0,
        randomize: bool = False,
        max_position_embeddings: int = 4096,
        config=None
    ):
        if config is None:
            config = model.config
        model = getattr(model, 'base_model', model)
        dim = config.hidden_size / config.num_attention_heads

        #
        logger.info("NTK Aware - Rotary Position Embedding hyperparameters")
        logger.info(f"Dimension: {dim}")
        logger.info(f"Scale (for the base): {scale}")
        logger.info(f"Scale Power: {scale_power}")
        logger.info(f"Max Pos Emb: {max_position_embeddings}")
        rotary_emb = NTKAwareRope(
            dim=dim,
            scale=scale,
            scale_power=scale_power,
            truncate=truncate,
            randomize=randomize,
            max_position_embeddings=max_position_embeddings,
            device=model.device,
        )
        for decoder in model.layers:
            assert hasattr(decoder, 'self_attn') and hasattr(decoder.self_attn, 'rotary_emb')
            assert decoder.self_attn.rotary_emb.inv_freq.shape == rotary_emb.freq_2pi.shape
            decoder.self_attn.rotary_emb = rotary_emb


        if randomize:
            try:
                from fastchat.train import llama_flash_attn_monkey_patch
                llama_flash_attn_monkey_patch.apply_rotary_pos_emb = batch_apply_rotary_pos_emb
            except ImportError:
                pass
            modeling_llama.saved_apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
            modeling_llama.apply_rotary_pos_emb = batch_apply_rotary_pos_emb

            def randomize_hook(module: torch.nn.Module, _: tuple):
                # Need kwargs to do this without the lazy caching with rebuild_random
                if module.training:
                    rotary_emb.rebuild_random = True
            rotary_emb.random_init_hook = model.register_forward_pre_hook(randomize_hook)
        return rotary_emb

class NTKAwareByParts(ScaledLlamaRotaryEmbedding):

    def __init__(
        self,
        dim: int,
        lower_bound=1,
        upper_bound=32,
        scale: float = 1, scale_power: float = 0, truncate: int = 0, randomize: bool = False, max_position_embeddings: int = 4096, base=10000, device=None
    ):
        #
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        #
        dims = torch.arange(0, dim, 2).float().to(device)
        wavelength = 2*PI * (base**(2 * dims / dim))
        num_rotations = max_position_embeddings / wavelength
        gamma = self.get_gamma(num_rotations=num_rotations)

        #
        freq_2pi = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))

        #
        freq_2pi_new = ((1 - gamma) * (freq_2pi / scale)) + (gamma * freq_2pi)

        super().__init__(
            base=base,
            freq_2pi=freq_2pi_new,
            max_position_embeddings=max_position_embeddings,
            dim=dim,
            scale=1, # 1 becuase we already accounted the effect of scale in freq_2pi_new
            scale_power=scale_power,
            device=device
        )

    def get_gamma(
        self,
        num_rotations,
    ):
        gamma = (num_rotations - self.lower_bound) / (self.upper_bound - self.lower_bound)
        # if the wavelength is way bigger than context length (ratio < self.lower_bound] = 0
        # if the wavelength is way smaller than context length (ratio > self.upper_bound) = 1
        gamma = torch.clip(gamma, min=0, max=1)

        return gamma

    @staticmethod
    def patch(
        model: torch.nn.Module,
        seq_len_train: int,
        scale: float = 1.0,
        scale_power: float = 0.0,
        truncate: int = 0,
        randomize: bool = False,
        max_position_embeddings: int = 4096,
        config=None
    ):
        if config is None:
            config = model.config
        model = getattr(model, 'base_model', model)
        dim = config.hidden_size / config.num_attention_heads

        #
        logger.info("NTK Aware by Parts - Rotary Position Embedding hyperparameters")
        logger.info(f"Dimension: {dim}")
        logger.info(f"Scale (for the base): {scale}")
        logger.info(f"Scale Power: {scale_power}")
        logger.info(f"Max Pos Emb: {max_position_embeddings}")
        # rotary_emb = NTKAwareByParts(
        #     seq_len_train=seq_len_train,
        #     seq_len_test=max_position_embeddings,
        #     dim=dim,
        #     lower_bound=1,
        #     upper_bound=32,
        #     scale=scale,
        #     scale_power=scale_power,
        #     truncate=truncate,
        #     randomize=randomize,
        #     max_position_embeddings=max_position_embeddings,
        #     device=model.device,
        # )
        rotary_emb = NTKAwareByParts(
            dim=dim,
            scale=scale,
            scale_power=scale_power,
            truncate=truncate,
            randomize=randomize,
            max_position_embeddings=max_position_embeddings,
            device=model.device,
        )
        for decoder in model.layers:
            assert hasattr(decoder, 'self_attn') and hasattr(decoder.self_attn, 'rotary_emb')
            assert decoder.self_attn.rotary_emb.inv_freq.shape == rotary_emb.freq_2pi.shape
            decoder.self_attn.rotary_emb = rotary_emb


        if randomize:
            try:
                from fastchat.train import llama_flash_attn_monkey_patch
                llama_flash_attn_monkey_patch.apply_rotary_pos_emb = batch_apply_rotary_pos_emb
            except ImportError:
                pass
            modeling_llama.saved_apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
            modeling_llama.apply_rotary_pos_emb = batch_apply_rotary_pos_emb

            def randomize_hook(module: torch.nn.Module, _: tuple):
                # Need kwargs to do this without the lazy caching with rebuild_random
                if module.training:
                    rotary_emb.rebuild_random = True
            rotary_emb.random_init_hook = model.register_forward_pre_hook(randomize_hook)
        return rotary_emb