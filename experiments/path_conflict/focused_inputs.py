"""Disjoint source roles and fixed candidate-wording controls for the same prompts."""

import numpy as np

from .data import token_text_offsets, quote_tokens
from .focused_plan import SOURCE_ROLES, TARGET_HEADS


def partition_sources(prompt_length, count, roles, recent=10):
    """History excludes the decision-query key. Its self write has its own group."""
    query = count - 1
    boundary = max(prompt_length, query - recent)
    groups = dict(roles)
    groups['query_self'] = np.array([query], dtype=int)
    groups['recent_history'] = np.arange(boundary, query, dtype=int)
    groups['remote_history'] = np.arange(prompt_length, boundary, dtype=int)
    used = np.concatenate(list(roles.values()) + [groups['query_self']])
    groups['other_prompt'] = np.setdiff1d(np.arange(prompt_length), used)
    return groups


def compile_roles(tokenizer, probe, case_id, recent):
    prompt = probe['prefix_ids'][:probe['prompt_length']]
    text, offsets = token_text_offsets(tokenizer, prompt)
    roles = {}
    for name, quotes in SOURCE_ROLES[case_id].items():
        for quote in quotes:
            assert text.count(quote) == 1, (case_id, name, 'ambiguous source quote')
        roles[name] = np.unique(np.concatenate([quote_tokens(text, offsets, quote) for quote in quotes]))
    groups = partition_sources(len(prompt), len(probe['prefix_ids']), roles, recent)
    # The identities are checked once when the experiment is compiled.
    joined = np.concatenate(list(groups.values()))
    np.testing.assert_array_equal(np.sort(joined), np.arange(len(probe['prefix_ids'])))
    return groups


def shared_prefix(candidates):
    common = 0
    for left, right in zip(*candidates):
        if left != right:
            break
        common += 1
    return common


def candidate_panels(tokenizer, native_probe, case, recent=10):
    """Natural wording plus a headwear determiner-controlled, forced-prefix panel.

    The singular panel is a new teacher-forced contrast, not another natural draw.
    Both alternatives append ' a'; the intervention then precedes cap/headdress.
    """
    variants = [('natural', case['candidates'])]
    if case['case_id'] == '14315_headwear_scope':
        variants.append(('parallel_singular', [
            ' a cap with a folded piece of cloth tied on top',
            ' a headdress with a special fringe of gold and feathers',
        ]))
    result = []
    native_prefix = native_probe['prefix_ids']
    text = tokenizer.decode(native_prefix, clean_up_tokenization_spaces=False)
    for panel, strings in variants:
        candidates = [tokenizer.encode(value, add_special_tokens=False) for value in strings]
        for string, tokens in zip(strings, candidates):
            decoded = tokenizer.decode(native_prefix + tokens, clean_up_tokenization_spaces=False)
            assert decoded == text + string, 'Candidate does not match its declared text'
        common = shared_prefix(candidates)
        assert common < min(map(len, candidates)), 'No contrasting continuation'
        probe = dict(native_probe, prefix_ids=native_prefix + candidates[0][:common],
                     candidates=[tokens[common:] for tokens in candidates], trace_heads=TARGET_HEADS,
                     trace_window=10, panel=panel, forced_common_tokens=common,
                     candidate_texts=strings, native_prefix_length=len(native_prefix))
        probe['groups'] = compile_roles(tokenizer, probe, case['case_id'], recent)
        result.append(probe)
    return result


def source_rows(tokenizer, probe, identity):
    rows = []
    for role, indices in probe['groups'].items():
        for source in indices:
            token = int(probe['prefix_ids'][source])
            rows.append(dict(identity, role=role, source=int(source), token_id=token,
                token_text=tokenizer.decode([token], clean_up_tokenization_spaces=False),
                query=len(probe['prefix_ids']) - 1, lag=len(probe['prefix_ids']) - 1 - int(source)))
    return rows
