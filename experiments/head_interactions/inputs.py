"""Input adapters end here; experiments see immutable prefixes and explicit source roles."""

from dataclasses import asdict, dataclass, field

from state_audit.dataset import Example
from state_audit.experiments.messages import MessageSite
from state_audit.pairing import load_answer
from state_audit.storage import read_json
from state_audit.tokenization import (
    encode_prompt,
    positions_for_quotes,
    quote_positions,
    special_token_ids,
)


@dataclass
class Case:
    id: str
    source_id: str
    prefix_ids: list[int]
    candidates: list[list[int]]
    sources: dict[str, tuple[int, ...]]
    dataset: str = "unknown"
    task: str = "unknown"
    generator: str = "unknown"
    metadata: dict = field(default_factory=dict)
    donor: str | None = None
    donor_kind: str | None = None
    preferred: int = 0


def compile_case(row, base, tokenizer, template):
    row = dict(row)
    prompt = None
    if "answer" in row:
        answer = load_answer(base / row.pop("answer"))
        prompt = answer
        target = row.pop("target")
        if not 0 <= target < len(answer["response_ids"]):
            raise ValueError("Claim target must be a saved response token")
        row["prefix_ids"] = answer["token_ids"][: answer["prompt_length"] + target]
        row["prompt_length"] = answer["prompt_length"]
        row.setdefault("source_id", answer["source_id"])
    elif "example" in row:
        example = Example(**row.pop("example"))
        encoded = encode_prompt(tokenizer, example, template)
        prompt = encoded
        prefix = tokenizer.encode(row.pop("response_prefix", ""), add_special_tokens=False)
        row["prefix_ids"] = encoded["prompt_ids"] + prefix
        row["prompt_length"] = len(encoded["prompt_ids"])
        row.setdefault("source_id", example.source_id)
        row.setdefault("sources", {})
        for index, span in enumerate(example.evidence):
            keys = [key for key, source in enumerate(encoded["key_sources"]) if source == index]
            row["sources"][span["id"]] = keys
    if prompt is not None:
        sources = row.setdefault("sources", {})
        for name, quotes in row.pop("source_quotes", {}).items():
            sources[name] = positions_for_quotes(
                prompt["prompt_text"], prompt["prompt_offsets"], quotes
            )
    return finish_case(row, tokenizer)


def finish_case(row, tokenizer):
    prefix = row["prefix_ids"]
    prompt_length = row.pop(
        "prompt_length", row.get("metadata", {}).get("prompt_length", len(prefix))
    )
    sources = row.setdefault("sources", {})
    for keys in sources.values():
        if any(key < 0 or key >= len(prefix) for key in keys):
            raise ValueError("Source positions must belong to the saved prefix")
    for name, quotes in row.pop("source_quotes", {}).items():
        sources[name] = quote_positions(tokenizer, prefix[:prompt_length], quotes)
    special = set(special_token_ids(tokenizer))
    ordinary = {index for index, token in enumerate(prefix) if token not in special}
    sources["ordinary"] = sorted(ordinary)
    sources["history"] = sorted(index for index in ordinary if index >= prompt_length)
    row["sources"] = {name: tuple(sorted(set(keys) & ordinary)) for name, keys in sources.items()}
    row["candidates"] = [
        tokenizer.encode(value, add_special_tokens=False) if isinstance(value, str) else value
        for value in row["candidates"]
    ]
    row.setdefault("metadata", {})["prompt_length"] = prompt_length
    return Case(**row)


def read_cases(path, tokenizer):
    document = read_json(path)
    template = document.get("template", "chat")
    cases = [compile_case(row, path.parent, tokenizer, template) for row in document["cases"]]
    validate_cases(cases)
    return cases


def read_paired_cases(root, tokenizer):
    """Reuse existing paired_head_transport exports; never import its model/hook code."""
    settings = read_json(root / "paired_config.json")
    cases = []
    for path in sorted((root / "pairs").glob("*/reviewed_case.json")):
        review = read_json(path)
        for side in ("supported", "unsupported"):
            context = read_json(path.parent / side / "onset/context.json")
            row = dict(
                id=review["case_id"] + ":" + side,
                source_id=str(review["source_id"]),
                prefix_ids=context["prefix_ids"],
                candidates=context["candidates"],
                prompt_length=len(context["prefix_ids"]) - context["position"],
                source_quotes=review["source_roles"],
                dataset="RAGTruth",
                task=review.get("task", "unknown"),
                generator=settings["sampling"]["model"],
                metadata=dict(
                    side=side,
                    history_status=review["history_status"][side],
                    support_scope=review["evidence_status"],
                    readout_status="different_attribute_slots"
                    if "onion_stage" in review["case_id"]
                    else "natural_wording_not_minimal_pair",
                    comparison="within_fixed_prefix; natural sides have different histories",
                ),
            )
            case = finish_case(row, tokenizer)
            case.sources["evidence"] = tuple(
                sorted(set(case.sources["scope"]) | set(case.sources["supported_value"]))
            )
            cases.append(case)
    validate_cases(cases)
    return cases


def validate_cases(cases):
    lookup = {case.id: case for case in cases}
    if not cases or len(lookup) != len(cases):
        raise ValueError("Audit cases must be nonempty and have unique IDs")
    for case in cases:
        if case.preferred not in (0, 1):
            raise ValueError(f"{case.id}: preferred must identify candidate 0 or 1")
        if case.donor is not None:
            lookup[case.donor]
            if not case.donor_kind:
                raise ValueError(f"{case.id}: describe the donor relation in donor_kind")


def compile_groups(plan, case, model, shift=0):
    groups = {}
    for group in plan["groups"]:
        if group["name"] in groups or not group["sites"]:
            raise ValueError("Group names must be unique and each group must contain sites")
        sites = []
        for index, row in enumerate(group["sites"]):
            positions = tuple(
                len(case.prefix_ids) - 1 + shift + offset for offset in row.get("offsets", [0])
            )
            source = row.get("source", "all")
            keys = None if source == "all" else case.sources[source]
            site = MessageSite(
                group["name"] + f":{index}", row["layer"], tuple(row["heads"]), positions, keys
            )
            validate_site(site, case, model)
            sites.append(site)
        groups[group["name"]] = sites
    validate_groups(groups)
    return groups


def validate_site(site, case, model):
    count, _, _ = model.head_layout(site.layer)
    if not site.heads or len(set(site.heads)) != len(site.heads):
        raise ValueError(f"{site.name}: heads must be nonempty and unique")
    if min(site.heads) < 0 or max(site.heads) >= count:
        raise ValueError(f"{site.name}: head outside native model layout")
    if not site.positions or min(site.positions) < 0 or max(site.positions) >= len(case.prefix_ids):
        raise ValueError(f"{site.name}: intervention must stay inside the fixed prefix")


def validate_groups(groups):
    sites = [site for values in groups.values() for site in values]
    if len({site.name for site in sites}) != len(sites) or not sites:
        raise ValueError("Groups and site names must be unique and nonempty")
    for index, left in enumerate(sites):
        for right in sites[index + 1 :]:
            overlap = left.layer == right.layer and set(left.heads) & set(right.heads)
            overlap = overlap and set(left.positions) & set(right.positions)
            if overlap and (
                left.keys is None or right.keys is None or set(left.keys) & set(right.keys)
            ):
                raise ValueError(f"Overlapping factorial messages: {left.name}, {right.name}")


def describe_case(case, groups, tokenizer):
    sites = [asdict(site) for values in groups.values() for site in values]
    positions = sorted({position for site in sites for position in site["positions"]})
    return dict(
        **asdict(case),
        sites=sites,
        prefix_text=tokenizer.decode(case.prefix_ids, skip_special_tokens=False),
        candidate_texts=[tokenizer.decode(ids) for ids in case.candidates],
        intervention_tokens=[
            dict(
                position=p, token_id=case.prefix_ids[p], text=tokenizer.decode([case.prefix_ids[p]])
            )
            for p in positions
        ],
    )
