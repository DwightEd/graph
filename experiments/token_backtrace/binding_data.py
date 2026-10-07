"""Content-fixed owner swaps with disjoint entity names and balanced fact order."""
from collections import Counter
from itertools import product


NAMES = """Alice Bob Carol David Emma Frank Grace Henry Irene Jack Kate Liam Maya
Noah Owen Paul Quinn Rose Sara Tom Vera Will Adam Beth Carl Dana Evan Finn Gina Hugh
Ivan Jane Kyle Lily Mark Nina Olga Peter Ruby Sean Tina Troy Wade Yara Zoe Alan Brad
Cora Dean Ella Gail Hope Jean Judy Luke Nora Ruth Seth Tara Alex Amy Ben Bill Blake
Clara Cole Dale Dave Eric Eva Felix Fred George Helen Ian James Jesse John Joy Julia
Laura Leo Lisa Louis Lucy Mary Max Mike Miles Nick Oliver Oscar Rachel Ray Sam Simon
Steve Susan Tim Tony Victor Walter Wendy William Zach""".split()


def encode_case(tokenizer, pair, pair_id, split, owner, order, style):
    """7 has fixed content; its owner is the queried canonical participant iff owner=0."""
    other = ("3", "5", "9")[pair_id % 3]
    owners = [pair[owner], pair[1 - owner]]
    facts = [f"Record: {owners[0]} {style} 7 marbles.",
             f"Record: {owners[1]} {style} {other} marbles."]
    if order:
        facts.reverse()
    content = (f"Participants: {pair[0]}, {pair[1]}.\n" + "\n".join(facts)
               + f"\nHow many marbles does {pair[0]} have? Respond with one digit only.")
    chat = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    value_char = rendered.index(" 7 marbles.") + 1
    key = next(i for i, (begin, end) in enumerate(encoded["offset_mapping"])
               if begin <= value_char < end)
    prompt = encoded["input_ids"]
    assert prompt == tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    assert tokenizer.decode([prompt[key]]) == "7"
    return dict(id=f"p{pair_id:02d}_{owner}_{order}_{style}", pair_id=pair_id,
                split=split, pair=pair, owner=owner, label=int(owner == 0), order=order,
                style=style, other=other, key=key, query=len(prompt) - 1,
                prompt=prompt, content=content)


def build_cases(tokenizer, counts=(12, 4, 8), reverse_participants=False):
    """Names never cross fit/dev/test; every pair has both owners, orders and styles."""
    names = [name for name in dict.fromkeys(NAMES)
             if len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1]
    assert len(names) >= 2 * sum(counts)
    cases = []
    first_pair = 0
    for split, count in zip(("fit", "dev", "test"), counts):
        for pair_id in range(first_pair, first_pair + count):
            pair = names[2 * pair_id:2 * pair_id + 2]
            canonical_orders = (0, 1) if reverse_participants else (0,)
            for reverse, owner, order, style in product(canonical_orders, range(2), range(2), ("owns", "has")):
                participants = pair[::-1] if reverse else pair
                case = encode_case(tokenizer, participants, pair_id, split, owner, order, style)
                case["id"] += f"_r{reverse}"
                case["canonical_order"] = reverse
                cases.append(case)
        first_pair += count
    validate_cases(cases)
    return cases


def matched_donor(cases, case, *, change_owner=False, change_style=False):
    owner = 1 - case["owner"] if change_owner else case["owner"]
    style = ("has" if case["style"] == "owns" else "owns") if change_style else case["style"]
    return next(i for i, donor in enumerate(cases)
                if (donor["pair_id"], donor["owner"], donor["order"], donor["style"])
                == (case["pair_id"], owner, case["order"], style)
                and donor["pair"] == case["pair"])


def validate_cases(cases):
    """Checks that owner swaps change assignment, not position or the token multiset."""
    names_by_split = {split: set() for split in ("fit", "dev", "test")}
    for case in cases:
        names_by_split[case["split"]].update(case["pair"])
        donor = cases[matched_donor(cases, case, change_owner=True)]
        assert case["key"] == donor["key"] and case["query"] == donor["query"]
        assert Counter(case["prompt"]) == Counter(donor["prompt"])
    assert not names_by_split["fit"] & names_by_split["test"]
    assert not names_by_split["dev"] & names_by_split["test"]
    assert not names_by_split["fit"] & names_by_split["dev"]
