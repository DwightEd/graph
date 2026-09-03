from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    file.write_text(text.replace(old, new), encoding="utf-8")


graph = "experiments/attention_mechanism_audit/graph.py"
replace_once(
    graph,
    '''PROFILE_CHANNELS = (
    "attention",
    "residual_message_norm",
    "positive_function",
    "negative_function",
)
STAGE_PRE, STAGE_ATTN, STAGE_OUT = range(3)
''',
    '''PROFILE_CHANNELS = (
    "attention",
    "residual_message_norm",
    "positive_function",
    "negative_function",
)
TOKEN_FLOW_CHANNELS = (
    "positive_function",
    "negative_function",
    "attention",
    "residual_message_norm",
)
STAGE_PRE, STAGE_ATTN, STAGE_OUT = range(3)
''',
)
replace_once(
    graph,
    '''    node_embedding: torch.Tensor
    edge_index: torch.Tensor
''',
    '''    node_embedding: torch.Tensor
    token_flow: torch.Tensor
    edge_index: torch.Tensor
''',
)
replace_once(
    graph,
    '''    target_logprob: torch.Tensor = field(init=False)
    target_margin: torch.Tensor = field(init=False)
    _source: list[torch.Tensor] = field(default_factory=list)
''',
    '''    target_logprob: torch.Tensor = field(init=False)
    target_margin: torch.Tensor = field(init=False)
    token_flow: torch.Tensor = field(init=False)
    _source: list[torch.Tensor] = field(default_factory=list)
''',
)
replace_once(
    graph,
    '''        self.mlp_profile = torch.zeros(response, self.layers, 3, dtype=torch.float32)
        self.target_logprob = torch.zeros(response, dtype=torch.float32)
        self.target_margin = torch.zeros(response, dtype=torch.float32)
''',
    '''        self.mlp_profile = torch.zeros(response, self.layers, 3, dtype=torch.float32)
        self.target_logprob = torch.zeros(response, dtype=torch.float32)
        self.target_margin = torch.zeros(response, dtype=torch.float32)
        self.token_flow = torch.zeros(
            response,
            self.token_count,
            len(TOKEN_FLOW_CHANNELS),
            dtype=torch.float32,
        )
''',
)
replace_once(
    graph,
    '''        dense = _profile(attention, transport, function, roles)
        self.profile[target, layer] = dense.cpu()

        priority = function.abs()
''',
    '''        dense = _profile(attention, transport, function, roles)
        self.profile[target, layer] = dense.cpu()
        source_count = attention.shape[1]
        self.token_flow[target, :source_count, 0] += (
            function.clamp_min(0).sum(dim=0).cpu()
        )
        self.token_flow[target, :source_count, 1] += (
            (-function).clamp_min(0).sum(dim=0).cpu()
        )
        self.token_flow[target, :source_count, 2] += attention.sum(dim=0).cpu()
        self.token_flow[target, :source_count, 3] += transport.sum(dim=0).cpu()

        priority = function.abs()
''',
)
replace_once(
    graph,
    '''            node_embedding=embedding,
            edge_index=edge_index,
''',
    '''            node_embedding=embedding,
            token_flow=self.token_flow,
            edge_index=edge_index,
''',
)

capture = "experiments/attention_mechanism_audit/capture.py"
replace_once(
    capture,
    '''    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    elif hasattr(cache, "key_cache"):
        cache = tuple(zip(cache.key_cache, cache.value_cache))
    return tuple((key.detach(), value.detach()) for key, value in cache)
''',
    '''    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    elif hasattr(cache, "layers"):
        cache = ((layer.keys, layer.values) for layer in cache.layers)
    elif hasattr(cache, "key_cache"):
        cache = zip(cache.key_cache, cache.value_cache)
    return tuple((key.detach(), value.detach()) for key, value in cache)
''',
)
replace_once(
    capture,
    '''    return DynamicCache.from_legacy_cache(tuple(layers))
''',
    '''    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(layers))
    return DynamicCache(tuple(layers))
''',
)
replace_once(
    capture,
    '''            "schema": "functional-message-graph-v1",
            "objective": "teacher_forced_target_logprob",
''',
    '''            "schema": "functional-message-graph-v2",
            "objective": "teacher_forced_target_logprob",
            "evidence_mask": evidence.detach().cpu(),
''',
)

collect = "experiments/attention_mechanism_audit/collect.py"
replace_once(collect, 'STATE_DIRECTORY = "functional_message_graph_v1"\n', 'STATE_DIRECTORY = "functional_message_graph"\n')
replace_once(
    collect,
    '''        "schema": "functional-message-graph-v1",
''',
    '''        "schema": "functional-message-graph-v2",
''',
)

capture_test = "experiments/attention_mechanism_audit/tests/test_capture.py"
replace_once(
    capture_test,
    '''    assert graph["schema"] == "functional-message-graph-v1"
    assert graph["node_profile"].shape[:3] == (3, 2, 4)
''',
    '''    assert graph["schema"] == "functional-message-graph-v2"
    assert graph["evidence_mask"].tolist() == [False, True, True, False]
    assert graph["node_profile"].shape[:3] == (3, 2, 4)
    assert graph["token_flow"].shape == (3, 7, 4)
''',
)

graph_test = "experiments/attention_mechanism_audit/tests/test_graph.py"
replace_once(
    graph_test,
    '''    PROFILE_CHANNELS,
    ROLE_NAMES,
''',
    '''    PROFILE_CHANNELS,
    ROLE_NAMES,
    TOKEN_FLOW_CHANNELS,
''',
)
replace_once(
    graph_test,
    '''    assert graph.edge_index.shape[1] <= 2
    assert graph.edge_head_message.shape == (graph.edge_index.shape[1], 3)

    selected = torch.zeros_like(graph.node_profile[0, 0].float())
''',
    '''    assert graph.edge_index.shape[1] <= 2
    assert graph.edge_head_message.shape == (graph.edge_index.shape[1], 3)
    assert graph.token_flow.shape == (3, 8, len(TOKEN_FLOW_CHANNELS))

    value_by_head = value[:, q_to_kv]
    head_gradient = torch.nn.functional.linear(
        gradient, output_weight.T
    ).reshape(2, 3)
    function = attention * torch.einsum(
        "shd,hd->hs", value_by_head, head_gradient
    )
    head_message = attention.T[..., None] * value_by_head
    transport = torch.einsum(
        "shd,hde,she->hs", head_message, block_gram, head_message
    ).clamp_min(0).sqrt()
    expected_flow = torch.stack(
        (
            function.clamp_min(0).sum(dim=0),
            (-function).clamp_min(0).sum(dim=0),
            attention.sum(dim=0),
            transport.sum(dim=0),
        ),
        dim=-1,
    )
    torch.testing.assert_close(graph.token_flow[0, :5], expected_flow)
    assert torch.count_nonzero(graph.token_flow[0, 5:]) == 0

    selected = torch.zeros_like(graph.node_profile[0, 0].float())
''',
)

readme = "experiments/attention_mechanism_audit/README.md"
replace_once(
    readme,
    '''Every causal source and every head contributes to `node_profile`:
''',
    '''Every causal source and every head contributes to `node_profile`.  The
same all-source atoms are also reduced only across layer/head into
`token_flow[response_target, source, channel]`, where the four channels are
positive function, negative function, attention, and exact residual-message
norm.  This compact temporal DAG preserves every source endpoint for downstream
flow algorithms without persisting every 128-dimensional head message.

Every causal source and every head contributes to `node_profile`:
''',
)
replace_once(
    readme,
    '''  functional_message_graph_v1/{train,test}/
''',
    '''  functional_message_graph/{train,test}/
''',
)

# A smoke limit keeps the same number of samples from each task, so the public
# all-task command never captures only the first task in dataset order.
replace_once(
    collect,
    '''    completed = {row["sample_id"] for row in rows}
    remaining = None if limit is None else max(limit - len(rows), 0)

    sources = load_sources(source_info)
''',
    '''    completed = {row["sample_id"] for row in rows}
    selected = {task: 0 for task in ("QA", "Summary", "Data2txt")}
    for row in rows:
        selected[task_name(row["task_type"])] += 1

    sources = load_sources(source_info)
''',
)
replace_once(
    collect,
    '''    written = 0

    for sample_id in dataset.sample_ids:
        sample_id = str(sample_id)
        if sample_id in completed:
            continue
        if remaining is not None and written >= remaining:
            break
        sample = dataset[sample_id]
        attention = sample.attention()
''',
    '''    for sample_id in dataset.sample_ids:
        if limit is not None and all(count >= limit for count in selected.values()):
            break
        sample_id = str(sample_id)
        if sample_id in completed:
            continue
        sample = dataset[sample_id]
        current_task = task_name(sample.task_type)
        if limit is not None and selected[current_task] >= limit:
            continue
        selected[current_task] += 1
        attention = sample.attention()
''',
)
replace_once(
    collect,
    '''            task_type=task_name(sample.task_type),
''',
    '''            task_type=current_task,
''',
)
replace_once(
    collect,
    '''        rows.append(row)
        written += 1

    complete = limit is None and len(rows) == len(dataset.sample_ids)
''',
    '''        rows.append(row)

    complete = limit is None and len(rows) == len(dataset.sample_ids)
''',
)

FILES = {
    'experiments/grounded_anchor_flow/__init__.py': '"""Grounded target-conditioned flow over exact functional message graphs."""\n\nfrom .flow import GroundedFlow, analyze_flow, flow_to_dict\n\n__all__ = ["GroundedFlow", "analyze_flow", "flow_to_dict"]\n',
    'experiments/grounded_anchor_flow/flow.py': '"""Source-resolved path flow on exact functional token messages.\n\nThe input is the all-source generation DAG saved by the functional-message\nobserver.  Edges are never learned and hallucination labels are never read.\n"""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Any, Mapping\n\nimport torch\n\nSOURCE_GROUPS = ("evidence", "other_prompt", "response")\nEVIDENCE, OTHER_PROMPT, RESPONSE = range(3)\nFLOW_CHANNELS = (\n    "positive_function",\n    "negative_function",\n    "attention",\n    "residual_message_norm",\n)\n\n\n@dataclass(frozen=True)\nclass GroundedFlow:\n    """Per-target path origins and response-anchor mediation."""\n\n    response_seeded_path_share: torch.Tensor  # [R]\n    response_seeded_anchor_flow: torch.Tensor  # [R]\n    source_path_posterior: torch.Tensor  # [R, evidence/question/response]\n    direct_response_share: torch.Tensor  # [R]\n    gather_distance: torch.Tensor  # [R]\n    anchor_occupancy: torch.Tensor  # [target, response anchor]\n    anchor_group_occupancy: torch.Tensor  # [target, source group, response anchor]\n    future_anchor_influence: torch.Tensor  # [response anchor]\n    anchor_concentration: torch.Tensor  # [target]\n    dominant_anchor: torch.Tensor  # [target], response index or -1\n    valid: torch.Tensor  # [R]\n    anchor_valid: torch.Tensor  # [R]\n\n\ndef token_transition(\n    token_flow: torch.Tensor,\n    response_start: int,\n    channel: str = "positive_function",\n) -> torch.Tensor:\n    """Normalize one capacity channel into a source-to-response transition.\n\n    The returned matrix has shape ``[all source tokens, response targets]``.\n    Column ``r`` contains the normalized incoming capacity of token\n    ``response_start + r``.  Non-causal cells and undefined columns are zero.\n    """\n\n    if channel not in FLOW_CHANNELS:\n        raise ValueError(f"unknown flow channel: {channel}")\n    flow = torch.as_tensor(token_flow, dtype=torch.float64)\n    if flow.ndim != 3 or flow.shape[-1] != len(FLOW_CHANNELS):\n        raise ValueError("token_flow must be [response, token, flow_channel]")\n    response, tokens, _ = flow.shape\n    if not 0 < response_start <= tokens or response != tokens - response_start:\n        raise ValueError("token_flow does not align with response_start")\n\n    target = torch.arange(response_start, tokens)[:, None]\n    source = torch.arange(tokens)[None]\n    capacity = flow[..., FLOW_CHANNELS.index(channel)].clamp_min(0)\n    capacity = capacity * (source < target)\n    total = capacity.sum(dim=1, keepdim=True)\n    normalized = torch.where(total > 0, capacity / total.clamp_min(1e-300), 0)\n    return normalized.T.contiguous()\n\n\ndef path_closure(response_transition: torch.Tensor) -> torch.Tensor:\n    """Sum every directed path product in a causal response DAG."""\n\n    if response_transition.ndim != 2 or response_transition.shape[0] != response_transition.shape[1]:\n        raise ValueError("response transition must be square")\n    if torch.count_nonzero(torch.tril(response_transition)):\n        raise ValueError("response transition must be strictly causal")\n    identity = torch.eye(\n        len(response_transition),\n        dtype=response_transition.dtype,\n        device=response_transition.device,\n    )\n    return torch.linalg.solve_triangular(\n        identity - response_transition,\n        identity,\n        upper=True,\n    )\n\n\ndef prompt_paths(\n    transition: torch.Tensor,\n    response_closure: torch.Tensor,\n    source_mask: torch.Tensor,\n) -> torch.Tensor:\n    """Average prompt-source path mass, retaining the common prompt prior."""\n\n    prompt_count = len(source_mask)\n    if not source_mask.any():\n        return torch.zeros(\n            response_closure.shape[1],\n            dtype=response_closure.dtype,\n            device=response_closure.device,\n        )\n    direct = transition[:prompt_count][source_mask].sum(dim=0) / prompt_count\n    return direct @ response_closure\n\n\ndef analyze_flow(\n    graph: Mapping[str, Any],\n    *,\n    channel: str = "positive_function",\n    gather_window: int = 64,\n    future_window: int = 64,\n) -> GroundedFlow:\n    """Trace prompt- and response-seeded paths through response anchors.\n\n    For targets with prior response tokens, a fixed binary prior assigns equal\n    total mass to prompt tokens and earlier response tokens, uniformly within\n    each side.  Conditioning this prior on reaching the target removes the\n    ordinary group-size advantage.  Response zero-hop starts are subtracted\n    before anchor mediation is measured, so a direct dependency is not\n    mislabeled as a multi-hop anchor route.\n    """\n\n    token_flow = torch.as_tensor(graph["token_flow"])\n    response_start = int(graph["response_start"])\n    evidence_mask = torch.as_tensor(graph["evidence_mask"], dtype=torch.bool)\n    response, tokens, _ = token_flow.shape\n    if evidence_mask.shape != (response_start,):\n        raise ValueError("evidence_mask must align with the prompt")\n    if gather_window <= 0 or future_window <= 0:\n        raise ValueError("flow windows must be positive")\n\n    transition = token_transition(token_flow, response_start, channel)\n    response_closure = path_closure(transition[response_start:])\n    evidence_paths = prompt_paths(transition, response_closure, evidence_mask)\n    question_paths = prompt_paths(transition, response_closure, ~evidence_mask)\n    response_prefix_paths = response_closure.cumsum(dim=0)\n\n    path_share = torch.full((response,), torch.nan, dtype=torch.float32)\n    anchor_flow = torch.full((response,), torch.nan, dtype=torch.float32)\n    source_posterior = torch.zeros(response, len(SOURCE_GROUPS), dtype=torch.float32)\n    direct_response = torch.full((response,), torch.nan, dtype=torch.float32)\n    gather_distance = torch.full((response,), torch.nan, dtype=torch.float32)\n    anchor_occupancy = torch.zeros(response, response, dtype=torch.float32)\n    anchor_group = torch.zeros(response, len(SOURCE_GROUPS), response, dtype=torch.float32)\n    future_anchor = torch.full((response,), torch.nan, dtype=torch.float32)\n    anchor_concentration = torch.full((response,), torch.nan, dtype=torch.float32)\n    dominant_anchor = torch.full((response,), -1, dtype=torch.int64)\n    valid = torch.zeros(response, dtype=torch.bool)\n    anchor_valid = torch.zeros(response, dtype=torch.bool)\n\n    for offset in range(response):\n        target = response_start + offset\n        incoming = transition[:target, offset]\n        if incoming.sum() > 0:\n            lag = target - torch.arange(target, dtype=torch.float64)\n            gather_distance[offset] = (\n                incoming * lag.clamp_max(gather_window)\n            ).sum().div(gather_window).float()\n            direct_response[offset] = incoming[response_start:target].sum().float()\n\n        if offset:\n            prompt_weight = response_weight = 0.5\n            response_paths = response_prefix_paths[offset - 1] / offset\n        else:\n            prompt_weight, response_weight = 1.0, 0.0\n            response_paths = torch.zeros(response, dtype=torch.float64)\n\n        forward = torch.stack(\n            (\n                prompt_weight * evidence_paths,\n                prompt_weight * question_paths,\n                response_weight * response_paths,\n            )\n        )\n        partition = forward[:, offset]\n        total_partition = partition.sum()\n        if total_partition <= 0:\n            continue\n\n        posterior = partition / total_partition\n        source_posterior[offset] = posterior.float()\n        path_share[offset] = posterior[RESPONSE].float()\n        valid[offset] = True\n\n        # Every target-reaching path that visits response node v factorizes as\n        # source->v times v->target.  Subtract paths whose sampled source is v\n        # itself; what remains is genuine transit through the anchor.\n        occupancy = (\n            forward[:, : offset + 1]\n            * response_closure[: offset + 1, offset]\n            / total_partition\n        )\n        source_start = torch.zeros_like(occupancy)\n        if response_weight:\n            source_start[RESPONSE, :offset] = (\n                (response_weight / offset)\n                * response_closure[:offset, offset]\n                / total_partition\n            )\n        transit = (occupancy - source_start).clamp_min(0)[:, :offset]\n        transit_mass = transit.sum()\n        if transit_mass <= 0:\n            continue\n\n        group_mass = transit.sum(dim=1)\n        anchor_flow[offset] = (group_mass[RESPONSE] / transit_mass).float()\n        per_anchor = transit.sum(dim=0)\n        normalized = per_anchor / transit_mass\n        anchor_occupancy[offset, :offset] = normalized.float()\n        anchor_group[offset, :, :offset] = (transit / transit_mass).float()\n        anchor_concentration[offset] = normalized.max().float()\n        dominant_anchor[offset] = int(normalized.argmax())\n        anchor_valid[offset] = True\n\n    for anchor in range(response):\n        stop = min(response, anchor + future_window + 1)\n        influence = anchor_occupancy[anchor + 1 : stop, anchor]\n        if len(influence):\n            future_anchor[anchor] = influence.mean()\n\n    return GroundedFlow(\n        response_seeded_path_share=path_share,\n        response_seeded_anchor_flow=anchor_flow,\n        source_path_posterior=source_posterior,\n        direct_response_share=direct_response,\n        gather_distance=gather_distance,\n        anchor_occupancy=anchor_occupancy,\n        anchor_group_occupancy=anchor_group,\n        future_anchor_influence=future_anchor,\n        anchor_concentration=anchor_concentration,\n        dominant_anchor=dominant_anchor,\n        valid=valid,\n        anchor_valid=anchor_valid,\n    )\n\n\ndef flow_to_dict(flow: GroundedFlow) -> dict[str, torch.Tensor]:\n    return {name: getattr(flow, name) for name in GroundedFlow.__dataclass_fields__}\n',
    'experiments/grounded_anchor_flow/pipeline.py': '"""Build source-resolved anchor-flow artifacts from functional message graphs."""\n\nfrom __future__ import annotations\n\nimport json\nfrom pathlib import Path\nfrom typing import Any\n\nimport torch\n\nfrom .flow import analyze_flow, flow_to_dict\n\nSTATE_DIRECTORY = "grounded_anchor_flow"\nCHANNELS = {\n    "functional": "positive_function",\n    "attention": "attention",\n    "message": "residual_message_norm",\n}\nCONTROL_FIELDS = (\n    "response_seeded_path_share",\n    "response_seeded_anchor_flow",\n    "direct_response_share",\n    "valid",\n    "anchor_valid",\n)\n\n\ndef read_index(root: Path) -> list[dict[str, Any]]:\n    path = root / "index.jsonl"\n    if not path.exists():\n        return []\n    return [\n        json.loads(line)\n        for line in path.read_text(encoding="utf-8").splitlines()\n        if line.strip()\n    ]\n\n\ndef analyze_graph(graph: dict[str, Any]) -> dict[str, Any]:\n    """Run one graph operator on functional capacity and two matched controls."""\n\n    if graph.get("schema") != "functional-message-graph-v2":\n        raise ValueError("grounded flow requires functional-message-graph-v2")\n    result: dict[str, Any] = {\n        "schema": "grounded-anchor-flow-v1",\n        "sample_id": str(graph["sample_id"]),\n        "source_id": str(graph["source_id"]),\n        "task_type": str(graph["task_type"]),\n        "generator_model": graph.get("generator_model"),\n        "response_start": int(graph["response_start"]),\n        "target_logprob": torch.as_tensor(graph["target_logprob"]).float(),\n        "labels_used": False,\n    }\n    functional = analyze_flow(graph, channel=CHANNELS["functional"])\n    result.update(\n        {f"functional_{name}": value for name, value in flow_to_dict(functional).items()}\n    )\n    for prefix in ("attention", "message"):\n        flow = flow_to_dict(analyze_flow(graph, channel=CHANNELS[prefix]))\n        result.update({f"{prefix}_{name}": flow[name] for name in CONTROL_FIELDS})\n    return result\n\n\ndef build_split(\n    graph_root: str | Path,\n    output_root: str | Path,\n    *,\n    limit: int | None = None,\n) -> dict[str, Any]:\n    graph_root = Path(graph_root)\n    output_root = Path(output_root)\n    sample_dir = output_root / "samples"\n    sample_dir.mkdir(parents=True, exist_ok=True)\n    rows = read_index(output_root)\n    completed = {str(row["sample_id"]) for row in rows}\n    graph_rows = read_index(graph_root)\n    written = 0\n\n    for row in graph_rows:\n        sample_id = str(row["sample_id"])\n        if sample_id in completed:\n            continue\n        if limit is not None and written >= limit:\n            break\n        graph = torch.load(\n            graph_root / row["path"], map_location="cpu", weights_only=False\n        )\n        artifact = analyze_graph(graph)\n        path = sample_dir / f"{sample_id}.pt"\n        temporary = path.with_suffix(".pt.tmp")\n        torch.save(artifact, temporary)\n        temporary.replace(path)\n        record = {\n            "sample_id": sample_id,\n            "source_id": artifact["source_id"],\n            "task_type": artifact["task_type"],\n            "generator_model": artifact["generator_model"],\n            "path": str(path.relative_to(output_root)),\n            "response_tokens": len(artifact["functional_response_seeded_path_share"]),\n        }\n        with (output_root / "index.jsonl").open("a", encoding="utf-8") as stream:\n            stream.write(json.dumps(record, sort_keys=True) + "\\n")\n        rows.append(record)\n        completed.add(sample_id)\n        written += 1\n\n    complete = limit is None and len(rows) == len(graph_rows)\n    manifest = {\n        "schema": "grounded-anchor-flow-v1",\n        "graph_root": str(graph_root.resolve()),\n        "samples": len(rows),\n        "complete": complete,\n        "labels_used": False,\n        "channels": CHANNELS,\n        "primary": "functional_response_seeded_anchor_flow",\n    }\n    (output_root / "manifest.json").write_text(\n        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"\n    )\n    return manifest\n\n\ndef build_all(\n    graph_state_root: str | Path,\n    output_root: str | Path,\n    *,\n    limit: int | None = None,\n) -> list[dict[str, Any]]:\n    return [\n        build_split(\n            Path(graph_state_root) / split,\n            Path(output_root) / STATE_DIRECTORY / split,\n            limit=limit,\n        )\n        for split in ("train", "test")\n    ]\n',
    'experiments/grounded_anchor_flow/evaluate.py': '"""Freeze anchor-flow scores, then open RAGTruth labels for evaluation."""\n\nfrom __future__ import annotations\n\nimport json\nfrom pathlib import Path\nfrom typing import Any, Iterable\n\nimport numpy as np\nimport torch\nfrom sklearn.metrics import average_precision_score, roc_auc_score\n\nPRIMARY = "functional_response_seeded_anchor_flow"\nSECONDARY = "functional_response_seeded_path_share"\nCONTROLS = (\n    "attention_response_seeded_anchor_flow",\n    "message_response_seeded_anchor_flow",\n    "functional_direct_response_share",\n    "attention_response_seeded_path_share",\n    "message_response_seeded_path_share",\n    "relative_response_position",\n    "response_length",\n    "target_surprisal",\n)\n\n\ndef read_index(root: Path) -> list[dict[str, Any]]:\n    return [\n        json.loads(line)\n        for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()\n        if line.strip()\n    ]\n\n\ndef position_adjust(score: np.ndarray, relative: np.ndarray, valid: np.ndarray) -> np.ndarray:\n    """Remove the ordinary response-position trend without labels."""\n\n    adjusted = np.full(len(score), np.nan, dtype=np.float32)\n    decile = np.minimum((relative * 10).astype(np.int16), 9)\n    for index in range(10):\n        selected = valid & np.isfinite(score) & (decile == index)\n        if not selected.any():\n            continue\n        value = score[selected]\n        center = np.median(value)\n        scale = 1.4826 * np.median(np.abs(value - center))\n        adjusted[selected] = (value - center) / (scale if scale >= 1e-6 else 1.0)\n    return adjusted\n\n\ndef freeze_scores(\n    inputs: Iterable[tuple[str | Path, str | Path]], task: str\n) -> tuple[dict[str, np.ndarray], list[tuple[Path, Path, dict[str, Any]]]]:\n    fields: dict[str, list[np.ndarray]] = {}\n    records: list[tuple[Path, Path, dict[str, Any]]] = []\n    for state_value, split_value in inputs:\n        state_root, split_root = Path(state_value), Path(split_value)\n        for row in read_index(state_root):\n            if str(row["task_type"]).casefold() != task.casefold():\n                continue\n            artifact = torch.load(\n                state_root / row["path"], map_location="cpu", weights_only=False\n            )\n            count = int(row["response_tokens"])\n            response_index = np.arange(count, dtype=np.int32)\n            values = {\n                "sample_id": np.repeat(str(row["sample_id"]), count),\n                "source_id": np.repeat(str(row["source_id"]), count),\n                "response_index": response_index,\n                "response_length": np.full(count, count, dtype=np.int32),\n                "relative_response_position": (response_index + 0.5) / count,\n                "target_surprisal": -artifact["target_logprob"].numpy(),\n                PRIMARY: artifact[PRIMARY].numpy(),\n                f"{PRIMARY}__valid": artifact["functional_anchor_valid"].numpy(),\n                SECONDARY: artifact[SECONDARY].numpy(),\n                f"{SECONDARY}__valid": artifact["functional_valid"].numpy(),\n                "functional_direct_response_share": artifact[\n                    "functional_direct_response_share"\n                ].numpy(),\n                "attention_response_seeded_anchor_flow": artifact[\n                    "attention_response_seeded_anchor_flow"\n                ].numpy(),\n                "attention_response_seeded_anchor_flow__valid": artifact[\n                    "attention_anchor_valid"\n                ].numpy(),\n                "message_response_seeded_anchor_flow": artifact[\n                    "message_response_seeded_anchor_flow"\n                ].numpy(),\n                "message_response_seeded_anchor_flow__valid": artifact[\n                    "message_anchor_valid"\n                ].numpy(),\n                "attention_response_seeded_path_share": artifact[\n                    "attention_response_seeded_path_share"\n                ].numpy(),\n                "attention_response_seeded_path_share__valid": artifact[\n                    "attention_valid"\n                ].numpy(),\n                "message_response_seeded_path_share": artifact[\n                    "message_response_seeded_path_share"\n                ].numpy(),\n                "message_response_seeded_path_share__valid": artifact[\n                    "message_valid"\n                ].numpy(),\n                "functional_gather_distance": artifact[\n                    "functional_gather_distance"\n                ].numpy(),\n                "functional_future_anchor_influence": artifact[\n                    "functional_future_anchor_influence"\n                ].numpy(),\n                "functional_anchor_concentration": artifact[\n                    "functional_anchor_concentration"\n                ].numpy(),\n            }\n            for name, value in values.items():\n                fields.setdefault(name, []).append(np.asarray(value))\n            records.append((state_root, split_root, row))\n    if not records:\n        raise ValueError(f"no {task} flow artifacts were found")\n\n    frozen = {name: np.concatenate(parts) for name, parts in fields.items()}\n    adjusted = f"{PRIMARY}_position_adjusted"\n    frozen[adjusted] = position_adjust(\n        frozen[PRIMARY],\n        frozen["relative_response_position"],\n        frozen[f"{PRIMARY}__valid"],\n    )\n    frozen[f"{adjusted}__valid"] = np.isfinite(frozen[adjusted])\n    return frozen, records\n\n\ndef load_labels(records: list[tuple[Path, Path, dict[str, Any]]]) -> np.ndarray:\n    from research_dataset import open_research_dataset\n\n    labels: list[np.ndarray] = []\n    by_split: dict[Path, list[dict[str, Any]]] = {}\n    for _state, split, row in records:\n        by_split.setdefault(split, []).append(row)\n    for split, rows in by_split.items():\n        dataset = open_research_dataset(\n            split, device="cpu", retain_embedded_labels=True\n        )\n        prepared = dataset.prepare_evaluation_labels(\n            [str(row["sample_id"]) for row in rows]\n        )\n        for row in rows:\n            sample = dataset[str(row["sample_id"])]\n            value = np.asarray(prepared.response_labels(sample).cpu(), dtype=bool)\n            sample.release_attention()\n            if len(value) != int(row["response_tokens"]):\n                raise ValueError("flow score and response label lengths differ")\n            labels.append(value)\n    return np.concatenate(labels)\n\n\ndef bootstrap_metrics(\n    label: np.ndarray,\n    first: np.ndarray,\n    source: np.ndarray,\n    replicates: int,\n    seed: int,\n    second: np.ndarray | None = None,\n) -> tuple[list[float | None], list[float | None], int]:\n    """Source-cluster intervals for one score or a paired score difference."""\n\n    groups = np.unique(source)\n    rows = {group: np.flatnonzero(source == group) for group in groups}\n    random = np.random.default_rng(seed)\n    result: list[tuple[float, float]] = []\n    for _ in range(replicates):\n        chosen = random.choice(groups, len(groups), replace=True)\n        index = np.concatenate([rows[group] for group in chosen])\n        if np.unique(label[index]).size != 2:\n            continue\n        auroc = roc_auc_score(label[index], first[index])\n        ap = average_precision_score(label[index], first[index])\n        if second is not None:\n            auroc -= roc_auc_score(label[index], second[index])\n            ap -= average_precision_score(label[index], second[index])\n        result.append((float(auroc), float(ap)))\n    if not result:\n        return [None, None], [None, None], 0\n    values = np.asarray(result)\n    return (\n        [float(x) for x in np.quantile(values[:, 0], (0.025, 0.975))],\n        [float(x) for x in np.quantile(values[:, 1], (0.025, 0.975))],\n        len(values),\n    )\n\n\ndef score_mask(name: str, arrays: dict[str, np.ndarray]) -> np.ndarray:\n    score = np.asarray(arrays[name], dtype=np.float64)\n    valid = np.isfinite(score)\n    if f"{name}__valid" in arrays:\n        valid &= np.asarray(arrays[f"{name}__valid"], dtype=bool)\n    return valid\n\n\ndef metric(\n    name: str,\n    arrays: dict[str, np.ndarray],\n    label: np.ndarray,\n    bootstrap: int,\n    seed: int,\n) -> dict[str, Any]:\n    score = np.asarray(arrays[name], dtype=np.float64)\n    valid = score_mask(name, arrays)\n    current_label, current_score = label[valid], score[valid]\n    result: dict[str, Any] = {\n        "tokens": int(valid.sum()),\n        "positives": int(current_label.sum()),\n        "auroc": None,\n        "average_precision": None,\n        "auroc_ci95": [None, None],\n        "average_precision_ci95": [None, None],\n        "bootstrap_successful": 0,\n    }\n    if np.unique(current_label).size != 2:\n        return result\n    result["auroc"] = float(roc_auc_score(current_label, current_score))\n    result["average_precision"] = float(\n        average_precision_score(current_label, current_score)\n    )\n    if bootstrap:\n        auroc_ci, ap_ci, successful = bootstrap_metrics(\n            current_label,\n            current_score,\n            arrays["source_id"][valid],\n            bootstrap,\n            seed,\n        )\n        result.update(\n            auroc_ci95=auroc_ci,\n            average_precision_ci95=ap_ci,\n            bootstrap_successful=successful,\n        )\n    return result\n\n\ndef paired_control(\n    primary: str,\n    control: str,\n    arrays: dict[str, np.ndarray],\n    label: np.ndarray,\n    bootstrap: int,\n    seed: int,\n) -> dict[str, Any]:\n    """Compare two capacities on the same tokens and source bootstrap draws."""\n\n    valid = score_mask(primary, arrays) & score_mask(control, arrays)\n    current_label = label[valid]\n    first = np.asarray(arrays[primary], dtype=np.float64)[valid]\n    second = np.asarray(arrays[control], dtype=np.float64)[valid]\n    result: dict[str, Any] = {\n        "control": control,\n        "tokens": int(valid.sum()),\n        "auroc_difference": None,\n        "average_precision_difference": None,\n        "auroc_difference_ci95": [None, None],\n        "average_precision_difference_ci95": [None, None],\n        "bootstrap_successful": 0,\n    }\n    if np.unique(current_label).size != 2:\n        return result\n    result["auroc_difference"] = float(\n        roc_auc_score(current_label, first) - roc_auc_score(current_label, second)\n    )\n    result["average_precision_difference"] = float(\n        average_precision_score(current_label, first)\n        - average_precision_score(current_label, second)\n    )\n    if bootstrap:\n        auroc_ci, ap_ci, successful = bootstrap_metrics(\n            current_label,\n            first,\n            arrays["source_id"][valid],\n            bootstrap,\n            seed,\n            second,\n        )\n        result.update(\n            auroc_difference_ci95=auroc_ci,\n            average_precision_difference_ci95=ap_ci,\n            bootstrap_successful=successful,\n        )\n    return result\n\n\ndef evaluate(\n    inputs: Iterable[tuple[str | Path, str | Path]],\n    task: str,\n    output: str | Path,\n    *,\n    bootstrap: int = 1000,\n    seed: int = 20260903,\n) -> dict[str, Any]:\n    arrays, records = freeze_scores(inputs, task)\n    output = Path(output)\n    output.parent.mkdir(parents=True, exist_ok=True)\n    np.savez_compressed(output.with_name("frozen_scores.npz"), **arrays)\n    label = load_labels(records)\n    adjusted = f"{PRIMARY}_position_adjusted"\n    names = (PRIMARY, adjusted, SECONDARY, *CONTROLS)\n    report = {\n        "task": task,\n        "samples": len(records),\n        "sources": int(np.unique(arrays["source_id"]).size),\n        "tokens": len(label),\n        "positives": int(label.sum()),\n        "prevalence": float(label.mean()),\n        "primary": PRIMARY,\n        "secondary": SECONDARY,\n        "score_meaning": (\n            "response-seeded share of target-conditioned flow through response "\n            "transit anchors; this is observed path attribution, not causal necessity"\n        ),\n        "metrics": {\n            name: metric(name, arrays, label, bootstrap, seed + index)\n            for index, name in enumerate(names)\n        },\n        "paired_capacity_controls": {\n            control: paired_control(\n                PRIMARY,\n                control,\n                arrays,\n                label,\n                bootstrap,\n                seed + 100 + index,\n            )\n            for index, control in enumerate(\n                (\n                    "attention_response_seeded_anchor_flow",\n                    "message_response_seeded_anchor_flow",\n                )\n            )\n        },\n        "labels_used_during": "evaluation_only_after_frozen_scores",\n    }\n    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")\n    np.savez_compressed(output.with_name("token_results.npz"), **arrays, label=label)\n    return report\n',
    'experiments/grounded_anchor_flow/run.py': '"""One foreground entry point for source-resolved functional anchor flow."""\n\nfrom __future__ import annotations\n\nimport argparse\nfrom pathlib import Path\n\nimport torch\n\nfrom experiments.attention_mechanism_audit.collect import (\n    STATE_DIRECTORY as GRAPH_DIRECTORY,\n    capture_all,\n)\nfrom .evaluate import CONTROLS, PRIMARY, SECONDARY, evaluate\nfrom .pipeline import STATE_DIRECTORY, build_all\n\nMODEL = Path(\n    "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct"\n)\nCACHE = Path(\n    "/share/home/tm902089733300000/a903202310/lys/research/"\n    "Unsupervised-hypergraph/outputs/attention_cache/"\n    "fresh_attention_c8847872bedf_20260731T074520Z_p876"\n)\nSOURCE_INFO = Path(\n    "/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl"\n)\nOUTPUT = Path(__file__).resolve().parent / "outputs" / MODEL.name\nTASKS = ("QA", "Summary", "Data2txt")\n\n\ndef state_pairs(state_root: Path, cache: Path):\n    return [\n        (state_root / STATE_DIRECTORY / split, cache / split)\n        for split in ("train", "test")\n    ]\n\n\ndef display(value):\n    return "n/a" if value is None else f"{value:.6f}"\n\n\ndef print_report(report):\n    print(f"\\n=== {report[\'task\'].upper()} GROUNDED ANCHOR FLOW ===")\n    print(\n        f"samples={report[\'samples\']} sources={report[\'sources\']} "\n        f"tokens={report[\'tokens\']} positives={report[\'positives\']} "\n        f"prevalence={report[\'prevalence\']:.4%}"\n    )\n    names = (\n        PRIMARY,\n        f"{PRIMARY}_position_adjusted",\n        SECONDARY,\n        *CONTROLS[:3],\n    )\n    for name in names:\n        value = report["metrics"][name]\n        print(\n            f"{name:52s} AUROC={display(value[\'auroc\'])} "\n            f"AP={display(value[\'average_precision\'])} tokens={value[\'tokens\']}"\n        )\n    for name, value in report["paired_capacity_controls"].items():\n        print(\n            f"paired functional - {name:39s} "\n            f"dAUROC={display(value[\'auroc_difference\'])} "\n            f"dAP={display(value[\'average_precision_difference\'])}"\n        )\n\n\ndef run_all(args):\n    graph_root = args.output / "graph"\n    capture_all(\n        args.cache,\n        args.source_info,\n        args.model,\n        graph_root,\n        device=args.device,\n        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype],\n        predictor_batch=args.predictor_batch,\n        edge_cover=args.edge_cover,\n        edge_budget=args.edge_budget,\n        limit=args.limit,\n    )\n    graph_state = graph_root / GRAPH_DIRECTORY\n    build_all(graph_state, args.output)\n    for task in TASKS:\n        report_path = args.output / "reports" / task.casefold() / "report.json"\n        report = evaluate(\n            state_pairs(args.output, args.cache),\n            task,\n            report_path,\n            bootstrap=args.bootstrap,\n            seed=args.seed,\n        )\n        print_report(report)\n        print(f"report: {report_path}")\n\n\ndef parser():\n    parser = argparse.ArgumentParser(\n        description="Source-resolved target-conditioned flow on exact AVWO messages"\n    )\n    parser.add_argument("--model", type=Path, default=MODEL)\n    parser.add_argument("--cache", type=Path, default=CACHE)\n    parser.add_argument("--source-info", type=Path, default=SOURCE_INFO)\n    parser.add_argument("--output", type=Path, default=OUTPUT)\n    parser.add_argument("--device", default="cuda:0")\n    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")\n    parser.add_argument("--predictor-batch", type=int, default=8)\n    parser.add_argument("--edge-cover", type=float, default=0.95)\n    parser.add_argument("--edge-budget", type=int, default=64)\n    parser.add_argument("--limit", type=int)\n    parser.add_argument("--bootstrap", type=int, default=1000)\n    parser.add_argument("--seed", type=int, default=20260903)\n    return parser\n\n\ndef main():\n    run_all(parser().parse_args())\n\n\nif __name__ == "__main__":\n    main()\n',
    'experiments/grounded_anchor_flow/run_all.sh': 'python -m experiments.grounded_anchor_flow.run "$@"\n',
    'experiments/grounded_anchor_flow/README.md': '# Grounded Anchor Flow\n\nThis experiment tests one mechanism instead of adding another stack of scalar\nattention features:\n\n> A correct continuation may pass through influential response anchors, but\n> those anchors remain reachable from the prompt/evidence.  A hallucinated\n> continuation may instead be carried by a response-seeded anchor backbone that\n> still has high downstream influence after prompt-seeded paths weaken.\n\nThe method combines two useful ideas without copying their attention-only\nimplementations.  The preplan-and-anchor work motivates separating long-range\nincoming gathering from later downstream influence.  FlowTracer motivates\nconditioning a causal DAG on reaching the current target and measuring the\ntransit nodes on all target-reaching paths.  Here both operations are performed\non the frozen model\'s real functional messages rather than on head-averaged raw\nattention.\n\n## 1. Exact generation edges\n\nFor response token `y_t`, the causal predictor is `q_t = P - 1 + t`.  At layer\n`l`, query head `h`, and source `s <= q_t`, the observer already records\n\n\\[\nm_{t,l,h,s}\n=\nW^O_{l,h}\\left(A^{l,h}_{q_t,s}V^{l,g(h)}_s\\right)\n\\]\n\nand its signed first-order support for the target log probability\n\n\\[\n\\phi_{t,l,h,s}\n=\n\\left\\langle\n\\frac{\\partial\\log p(y_t)}{\\partial o^l_{q_t}},\n m_{t,l,h,s}\n\\right\\rangle.\n\\]\n\nAll source, layer, and head atoms are retained in the original functional graph.\nFor global flow, they are reduced only over layer and head while source and\nresponse-target identities stay intact:\n\n\\[\nC^+_{s,t}=\\sum_{l,h}[\\phi_{t,l,h,s}]_+.\n\\]\n\nThe same all-source table stores veto, raw attention, and exact residual-message\nnorm.  Thus the identical graph algorithm can be run on:\n\n1. positive functional `AVWO` support — the method;\n2. raw attention — information-selection control;\n3. residual-message norm — information-transport control.\n\n## 2. Target-conditioned all-path flow\n\nEach response target column is normalized:\n\n\\[\nW_{s,t}=\\frac{C^+_{s,t}}{\\sum_{u<t}C^+_{u,t}}.\n\\]\n\nLet `B` be the response-to-response block of `W`.  It is strictly upper\ntriangular, so every response path is summed exactly by\n\n\\[\nH=(I-B)^{-1}=I+B+B^2+\\cdots.\n\\]\n\n`H[i,t]` is the total product weight of all response paths from response token\n`i` to target `t`.  Prompt-to-target paths are the direct prompt block followed\nby `H`; no full token-by-token matrix inverse is constructed.\n\nFor targets that have prior response tokens, a fixed source prior assigns half\nof its mass uniformly to the prompt and half uniformly to prior response\npositions.  Conditioning that prior on reaching the target gives the\n`response_seeded_path_share`.  This is a global path quantity; the ordinary\none-step `direct_response_share` is reported separately.\n\n## 3. Anchor mediation\n\nFor source group `g`, response transit token `v`, and target `t`, all paths that\npass through `v` factorize as\n\n\\[\nO_g(v\\mid t)\n\\propto\nH_g(v)H(v,t).\n\\]\n\nA prior response token can also be sampled as a zero-hop starting point.  That\nstart contribution is subtracted before anchor mediation is measured.  The\nremaining occupancy therefore represents genuine transit through `v`, not the\ntrivial fact that `v` is itself a response token.\n\nThe single primary score is\n\n\\[\n\\boxed{\n\\operatorname{RSAF}_t\n=\n\\frac{\\sum_{v<t}O_R(v\\mid t)}\n{\\sum_{g\\in\\{E,Q,R\\}}\\sum_{v<t}O_g(v\\mid t)}\n}\n\\]\n\nand is stored as `functional_response_seeded_anchor_flow`.  A high value means\nthat the response anchors mediating the current target are reached mainly from\nprior response seeds rather than prompt seeds.  It is a candidate\nself-confirming-flow signal, not a claim of semantic falsity or causal\nnecessity.\n\nThe full functional artifact additionally retains:\n\n- `source_path_posterior`: evidence, other-prompt, and response path origins;\n- `anchor_occupancy`: which earlier response tokens mediate each target;\n- `anchor_group_occupancy`: the source group of each anchor route;\n- `gather_distance`: clipped distance of incoming functional gathering;\n- `future_anchor_influence`: later target-conditioned occupancy through a token;\n- `dominant_anchor` and `anchor_concentration` for visualization.\n\n`gather_distance` and `future_anchor_influence` test the gather–anchor rhythm.\nThey are not multiplied into the detector.\n\n## 4. Evaluation contract\n\nHallucination labels are opened only after every graph and score has been saved.\nQA, Summary, and Data2txt are evaluated separately with token AUROC, average\nprecision, and source-cluster bootstrap intervals.\n\nThe primary functional RSAF is compared on the same valid tokens against:\n\n- attention-derived anchor flow;\n- message-norm-derived anchor flow;\n- response-seeded all-path share without requiring anchor mediation;\n- direct response dependence;\n- response position, response length, and target surprisal.\n\nA label-free response-position adjustment is reported because ordinary\nautoregressive generation naturally becomes more response-dependent over time.\n\nThe current experiment is an observed functional-flow audit.  A later\nconfirmatory phase must remove or keep the selected anchor backbone in a real\nmodel re-forward, with downstream gates recomputed, before claiming causal\nnecessity or sufficiency.\n\n## 5. Files\n\n```text\nflow.py      target-normalized path closure and anchor occupancy\npipeline.py  run the same graph operator on functional and control capacities\nevaluate.py  freeze scores, then load labels and compare capacities\nrun.py       one foreground end-to-end entry\nrun_all.sh   one-command launcher\ntests/       path, relay, anchor, control, and evaluation invariants\n```\n\nThe exact functional graph remains in the sibling\n`attention_mechanism_audit` package.  This package adds only the global graph\noperator and its evaluation; it does not introduce a learned GNN, autoencoder,\nor feature combiner.\n\n## Run\n\n```bash\nbash experiments/grounded_anchor_flow/run_all.sh\n```\n\nSmoke run:\n\n```bash\nbash experiments/grounded_anchor_flow/run_all.sh --limit 2 --bootstrap 0\n```\n',
    'experiments/grounded_anchor_flow/tests/__init__.py': '',
    'experiments/grounded_anchor_flow/tests/test_flow.py': 'import torch\n\nfrom experiments.grounded_anchor_flow.flow import (\n    EVIDENCE,\n    RESPONSE,\n    analyze_flow,\n    path_closure,\n    token_transition,\n)\n\n\ndef graph_from_support(support: torch.Tensor, response_start: int, evidence_mask):\n    response, tokens = support.shape\n    token_flow = torch.zeros(response, tokens, 4)\n    token_flow[..., 0] = support\n    token_flow[..., 2] = support\n    token_flow[..., 3] = support\n    return {\n        "token_flow": token_flow,\n        "response_start": response_start,\n        "evidence_mask": torch.tensor(evidence_mask, dtype=torch.bool),\n    }\n\n\ndef test_transition_normalizes_targets_and_closure_sums_all_paths():\n    support = torch.tensor(\n        [\n            [2.0, 1.0, 0.0, 0.0],\n            [1.0, 0.0, 3.0, 0.0],\n        ]\n    )\n    transition = token_transition(support[..., None].expand(-1, -1, 4), 2)\n    response = transition[2:]\n    closure = path_closure(response)\n\n    torch.testing.assert_close(transition[:2, 0].sum(), torch.tensor(1.0, dtype=torch.float64))\n    torch.testing.assert_close(transition[:3, 1].sum(), torch.tensor(1.0, dtype=torch.float64))\n    assert torch.count_nonzero(torch.tril(response)) == 0\n    expected = torch.tensor([[1.0, 0.75], [0.0, 1.0]], dtype=torch.float64)\n    torch.testing.assert_close(closure, expected)\n\n    response = torch.tensor(\n        [[0.0, 0.5, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.0]],\n        dtype=torch.float64,\n    )\n    closure = path_closure(response)\n    torch.testing.assert_close(closure[0, 2], torch.tensor(0.125, dtype=torch.float64))\n\n\ndef test_prompt_relay_reaches_anchor_without_becoming_response_seeded_transit():\n    # evidence token 0 -> response anchor 0 -> response target 1\n    support = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])\n    result = analyze_flow(graph_from_support(support, 1, [True]), future_window=4)\n\n    assert result.valid.tolist() == [True, True]\n    assert result.anchor_valid.tolist() == [False, True]\n    torch.testing.assert_close(result.source_path_posterior[1, EVIDENCE], torch.tensor(0.5))\n    torch.testing.assert_close(result.source_path_posterior[1, RESPONSE], torch.tensor(0.5))\n    torch.testing.assert_close(result.response_seeded_anchor_flow[1], torch.tensor(0.0))\n    assert result.dominant_anchor[1].item() == 0\n\n\ndef test_response_seeded_multihop_path_is_detected_at_the_anchor():\n    # response0 -> response1 -> response2, with no prompt route to the target.\n    support = torch.tensor(\n        [\n            [0.0, 0.0, 0.0, 0.0],\n            [0.0, 1.0, 0.0, 0.0],\n            [0.0, 0.0, 1.0, 0.0],\n        ]\n    )\n    result = analyze_flow(graph_from_support(support, 1, [True]), future_window=4)\n\n    assert result.valid[2]\n    assert result.anchor_valid[2]\n    torch.testing.assert_close(result.response_seeded_path_share[2], torch.tensor(1.0))\n    torch.testing.assert_close(result.response_seeded_anchor_flow[2], torch.tensor(1.0))\n    assert result.dominant_anchor[2].item() == 1\n    assert result.future_anchor_influence[1] > 0\n\n\ndef test_direct_response_dependency_is_not_anchor_mediation():\n    support = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])\n    result = analyze_flow(graph_from_support(support, 1, [True]))\n\n    assert result.valid[1]\n    assert not result.anchor_valid[1]\n    torch.testing.assert_close(result.response_seeded_path_share[1], torch.tensor(1.0))\n    assert torch.isnan(result.response_seeded_anchor_flow[1])\n\n\ndef test_balanced_prior_does_not_reward_a_group_for_having_more_tokens():\n    # Two prompt tokens and one response token have identical direct paths.\n    support = torch.tensor(\n        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]\n    )\n    result = analyze_flow(graph_from_support(support, 2, [True, False]))\n\n    torch.testing.assert_close(result.response_seeded_path_share[1], torch.tensor(0.5))\n    torch.testing.assert_close(result.source_path_posterior[1, :2].sum(), torch.tensor(0.5))\n',
    'experiments/grounded_anchor_flow/tests/test_pipeline.py': 'import torch\n\nfrom experiments.grounded_anchor_flow.pipeline import analyze_graph\n\n\ndef test_pipeline_keeps_full_functional_maps_and_compact_capacity_controls():\n    token_flow = torch.zeros(3, 4, 4)\n    token_flow[0, 0, 0] = 1.0\n    token_flow[1, 1, 0] = 1.0\n    token_flow[2, 2, 0] = 1.0\n    token_flow[..., 2] = token_flow[..., 0]\n    token_flow[..., 3] = token_flow[..., 0]\n    graph = {\n        "schema": "functional-message-graph-v2",\n        "sample_id": "sample",\n        "source_id": "source",\n        "task_type": "QA",\n        "generator_model": "generator",\n        "response_start": 1,\n        "evidence_mask": torch.tensor([True]),\n        "target_logprob": torch.tensor([-1.0, -2.0, -3.0]),\n        "token_flow": token_flow,\n    }\n\n    result = analyze_graph(graph)\n\n    assert result["labels_used"] is False\n    assert result["schema"] == "grounded-anchor-flow-v1"\n    assert result["functional_response_seeded_path_share"].shape == (3,)\n    assert result["functional_anchor_occupancy"].shape == (3, 3)\n    assert result["functional_source_path_posterior"].shape == (3, 3)\n    assert result["attention_response_seeded_anchor_flow"].shape == (3,)\n    assert "attention_anchor_occupancy" not in result\n    assert "message_source_path_posterior" not in result\n    assert result["functional_anchor_valid"].tolist() == [False, True, True]\n',
    'experiments/grounded_anchor_flow/tests/test_evaluate.py': 'import numpy as np\n\nfrom experiments.grounded_anchor_flow.evaluate import (\n    bootstrap_metrics,\n    paired_control,\n    position_adjust,\n)\n\n\ndef test_position_adjustment_removes_decile_center_without_labels():\n    score = np.asarray([1.0, 3.0, 10.0, 14.0])\n    relative = np.asarray([0.05, 0.05, 0.15, 0.15])\n    adjusted = position_adjust(score, relative, np.ones(4, dtype=bool))\n\n    np.testing.assert_allclose(np.median(adjusted[:2]), 0.0)\n    np.testing.assert_allclose(np.median(adjusted[2:]), 0.0)\n\n\ndef test_paired_control_uses_the_same_tokens_and_source_draws():\n    label = np.asarray([False, True, False, True, False, True])\n    arrays = {\n        "source_id": np.asarray(["a", "a", "b", "b", "c", "c"]),\n        "primary": np.asarray([0.0, 1.0, 0.1, 0.9, 0.2, 0.8]),\n        "primary__valid": np.asarray([True, True, True, True, True, False]),\n        "control": np.asarray([1.0, 0.0, 0.2, 0.8, 0.1, 0.9]),\n        "control__valid": np.asarray([True, True, True, True, False, True]),\n    }\n\n    result = paired_control("primary", "control", arrays, label, 50, 7)\n\n    assert result["tokens"] == 4\n    assert result["auroc_difference"] > 0\n    assert result["bootstrap_successful"] == 50\n\n\ndef test_empty_bootstrap_returns_undefined_intervals():\n    label = np.asarray([False, False])\n    score = np.asarray([0.0, 1.0])\n    source = np.asarray(["a", "b"])\n\n    auroc, ap, successful = bootstrap_metrics(label, score, source, 10, 3)\n\n    assert auroc == [None, None]\n    assert ap == [None, None]\n    assert successful == 0\n',
}
for relative, content in FILES.items():
    destination = Path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
