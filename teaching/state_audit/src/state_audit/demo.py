"""An offline, random-weight model for teaching and software checks only."""

import json
from dataclasses import asdict
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    MistralConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    set_seed,
)

from .dataset import Example, locate_evidence


def demo_examples() -> list[Example]:
    facts = ["Mira wears a blue coat.", "Sol wears a red coat."]
    prompt = "Evidence:\n" + "\n".join(facts) + "\nQuestion: What does Mira wear?"
    evidence = locate_evidence(prompt, facts)
    good = "Mira wears a blue coat. Sol wears a red coat."
    bad = "Mira wears a red coat. Sol wears a red coat."
    start = bad.index("red coat")
    labels = [dict(start=start, end=start + len("red coat"), text="red coat")]
    return [
        Example("good", "coat", prompt, evidence, good, [], dict(synthetic=True)),
        Example("bad", "coat", prompt, evidence, bad, labels, dict(synthetic=True)),
    ]


def build_demo(root: Path, family: str) -> tuple[Path, Path]:
    examples = demo_examples()
    data = root / "examples.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    data.write_text("".join(json.dumps(asdict(row)) + "\n" for row in examples), encoding="utf-8")
    words = set()
    splitter = pre_tokenizers.Whitespace()
    for text in ["User: Assistant:"] + [row.prompt + " " + row.response for row in examples]:
        words.update(token for token, _ in splitter.pre_tokenize_str(text))
    vocabulary = {
        word: index for index, word in enumerate(["[PAD]", "[UNK]", "[EOS]"] + sorted(words))
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = splitter
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "User: {{ messages[0]['content'] }}\nAssistant: "
    configs = {"llama": LlamaConfig, "mistral": MistralConfig, "qwen2": Qwen2Config}
    config = configs[family](
        vocab_size=len(vocabulary),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    set_seed(7)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model_path = root / "model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return data, model_path
