import functools
import gc
import inspect
import torch
from torch import Tensor
from typing import Callable, Optional

INIT_CHARS = [
    ".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}",
    "@", "#", "$", "%", "&", "*",
    "w", "x", "y", "z",
]

def get_nonascii_toks(tokenizer, device="cpu"):
    """Get token IDs for non-ASCII characters"""
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(nonascii_toks, device=device)

def mellowmax(t: Tensor, alpha=1.0, dim=-1):
    """Mellowmax function"""
    return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)))

def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Check if the exception is related to out-of-memory issues
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

def find_executable_batch_size(function: Callable = None, starting_batch_size: int = 128):
    """
    Decorator to automatically adjust batch size
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())

        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator

def configure_pad_token(tokenizer) -> None:
    """Configure padding token"""
    if tokenizer.pad_token:
        return

    if tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})