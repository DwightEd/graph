import re
from types import SimpleNamespace

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from experiments.unsupervised_token_graph.head_roles import binding
from experiments.unsupervised_token_graph.head_roles.tests.test_roles import tiny_llama


def tokenizer_fixture():
    text = "Use only these statements. likes the color What does like Answer with only. Answer:"
    text += " " + " ".join(binding.NAMES) + " " + " ".join(binding.COLORS)
    vocabulary = ["[UNK]", "[BOS]", *sorted(set(re.findall(r"\w+|[^\w\s]", text)))]
    backend = Tokenizer(models.WordLevel({word: index for index, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]")
    tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}\nAnswer:"
    return tokenizer


def test_full_input_binding_swaps_keep_facts_with_dynamic_intervals(tmp_path, monkeypatch):
    monkeypatch.setattr(binding, "NAMES", ("Alice Mary", *binding.NAMES[1:]))
    tokenizer = tokenizer_fixture()
    order = np.arange(8)
    original, blocks = binding.binding_input(tokenizer, order, 0)
    order[[0, 7]] = order[[7, 0]]
    changed, moved = binding.binding_input(tokenizer, order, 0)
    np.testing.assert_array_equal(np.sort(original), np.sort(changed))
    assert blocks[0, 1] - blocks[0, 0] != blocks[7, 1] - blocks[7, 0]
    np.testing.assert_array_equal(original[slice(*blocks[0])], changed[slice(*moved[7])])
    np.testing.assert_array_equal(original[slice(*blocks[7])], changed[slice(*moved[0])])
    monkeypatch.setattr(binding, "load_model", lambda args: tiny_llama())
    monkeypatch.setattr(binding.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    args = SimpleNamespace(output=tmp_path, seed=17, swaps=2, resume=False,
                           tokenizer="fixture", temperature=.05)
    rows = binding.run_binding(args, tokenizer.all_special_ids)
    assert len(rows) == 3 * 2 * 4
    assert all(np.isfinite(row["gap"]) for row in rows)
    with np.load(tmp_path / "binding/target_0.npz", allow_pickle=False) as saved:
        assert saved["L0__before"].shape == (2, 4, 2)
        assert saved["L0__after"].shape == (2, 4, 2)
