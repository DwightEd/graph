from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.reanchor_flow.worlds import (
    PairedWorld,
    SourceUnits,
    TargetContrast,
)


def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(17)
    config = LlamaConfig(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def paired_world() -> PairedWorld:
    clean = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    corrupt = torch.tensor([1, 8, 9, 4, 5, 6, 7])
    units = SourceUnits(
        torch.tensor([0, 1, 2, 0, 3, 4]),
        ("other", "passage:1", "passage:2", "response:4", "response:5"),
        ("other_prompt", "passage", "passage", "response", "response"),
    )
    target = TargetContrast(
        query_position=5,
        positive_token_id=7,
        negative_token_id=10,
        origin="controlled clean-vs-corrupt fact",
    )
    return PairedWorld(
        sample_id="tiny",
        tokenizer_id="tiny-llama",
        corruption="same-length replacement",
        clean_token_ids=clean,
        corrupt_token_ids=corrupt,
        response_start=4,
        units=units,
        candidate_unit_id=(1, 2),
        targets=(target,),
    ).check()
