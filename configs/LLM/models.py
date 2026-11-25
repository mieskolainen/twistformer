# Causal backbones and XGBoost score blending
#
# mikael.mieskolainen@cern.ch (2026)

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, Mamba2Config, Mamba2ForCausalLM

PAD, BOS, EOS = 0, 1, 2


def backbone(config, name, vocabulary, length):
    """
    Build the configured causal backbone from random weights

    Args:
        config:      Benchmark steering
        name:        Transformer or state-space model key
        vocabulary:  Size of the shared token vocabulary
        length:      Context length in tokens
    Returns:
        model:  Hugging Face causal language model
    """
    settings = config["models"][name]
    width, layers = settings["width"], settings["layers"]
    if name == "transformer":
        return GPT2LMHeadModel(GPT2Config(
            vocab_size=vocabulary, n_positions=length, n_embd=width, n_layer=layers, n_head=settings["heads"],
            resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD,
            use_cache=False,
        ))
    head_dim = settings["head_dim"]
    return Mamba2ForCausalLM(Mamba2Config(
        vocab_size=vocabulary, hidden_size=width, num_hidden_layers=layers, head_dim=head_dim,
        num_heads=2*width//head_dim, expand=2, n_groups=1, state_size=settings["state_size"],
        chunk_size=settings["chunk_size"], tie_word_embeddings=True, use_cache=False,
        bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD,
    ))


def expert_margins(experts, rows):
    """Return one raw tree margin per expert and token candidate"""
    return np.stack([
        booster.inplace_predict(rows[name].reshape(-1, rows[name].shape[-1]), predict_type="margin")
        .reshape(rows[name].shape[:-1])
        for name,booster in experts.items()
    ], axis=-1)


def adjust(logp, indices, scores, correction):
    """Reweight the shortlist while preserving its total probability mass"""
    shifted = scores + torch.as_tensor(correction,dtype=logp.dtype,device=logp.device)
    return logp.scatter(-1, indices, F.log_softmax(shifted,dim=-1)+torch.logsumexp(scores,dim=-1,keepdim=True))
