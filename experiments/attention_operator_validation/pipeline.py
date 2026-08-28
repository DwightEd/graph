"""Three-stage operator-code mechanism validation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from experiment_protocol import FrozenFile, dataset_manifest_sha256
from research_dataset import open_research_dataset
from experiments.grounded_route.config import GraphConfig
from experiments.grounded_route.graph import build_graph
from experiments.grounded_route.pipeline import select_samples

from .artifacts import (
    FeatureTable,
    implementation_sha256,
    load_feature_table,
    save_feature_table,
)
from .evaluate import (
    frozen_evaluation_spec,
    grouped_probe_report,
    univariate_report,
    validate_frozen_evaluation_spec,
    validate_label_free_bindings,
)
from .features import OPERATOR_MODES, extract_answer_features
from .operators import (
    extract_operator_geometry,
    file_sha256,
    load_operator_geometry,
    save_operator_geometry,
)
from .pair_codes import build_pair_code_field


def extract_operators(
    model_path,
    output_path,
    *,
    device: str = "cpu",
    load_dtype: str = "bfloat16",
    compute_dtype: str = "float32",
    block_heads: int = 4,
    trust_remote_code: bool = False,
    basis_dir=None,
) -> dict[str, object]:
    """Extract and freeze operator Gram matrices once for later reuse."""

    geometry = extract_operator_geometry(
        model_path,
        device=device,
        load_dtype=load_dtype,
        compute_dtype=compute_dtype,
        block_heads=block_heads,
        trust_remote_code=trust_remote_code,
        basis_dir=basis_dir,
    )
    save_operator_geometry(output_path, geometry)
    return {
        "operators": str(Path(output_path).resolve()),
        "sha256": file_sha256(output_path),
        "model_path": geometry.model_path,
        "architecture": geometry.architecture,
        "layers": geometry.layer_count,
        "heads": geometry.head_count,
        "kv_heads": geometry.kv_head_count,
        "head_dim": geometry.head_dim,
        "basis_dir": str(Path(basis_dir).resolve()) if basis_dir else None,
    }


def extract_features(
    data_root,
    operator_path,
    output_path,
    *,
    task: str = "QA",
    limit: int | None = None,
    imputation: str = "zero",
    seed: int = 20260828,
    graph_config: GraphConfig | None = None,
) -> dict[str, object]:
    """Freeze answer-level mechanism features without opening hallucination labels."""

    graph_config = GraphConfig() if graph_config is None else graph_config
    dataset = open_research_dataset(
        data_root,
        device="cpu",
        verify_hashes=True,
        retain_embedded_labels=False,
    )
    frozen_operator = FrozenFile.capture(operator_path)
    geometry = load_operator_geometry(frozen_operator.path)
    frozen_operator.verify(frozen_operator.path)
    expected = (
        int(dataset.manifest["num_layers"]),
        int(dataset.manifest["num_heads"]),
    )
    if expected != (geometry.layer_count, geometry.head_count):
        raise ValueError(
            "operator geometry does not match the attention-cache layer/head geometry"
        )
    sample_ids = select_samples(dataset, task, limit)
    if not sample_ids:
        raise ValueError("feature extraction selected no answers")

    rows: list[dict[str, float]] = []
    sources = []
    tasks = []
    lengths = []
    for sample_id in tqdm(sample_ids, desc="operator-code features", unit="answer"):
        sample = dataset[sample_id]
        try:
            graph = build_graph(sample, graph_config)
            field = build_pair_code_field(
                graph,
                imputation=imputation,
                include_self=True,
            )
            rows.append(
                extract_answer_features(
                    graph,
                    field,
                    geometry,
                    seed=seed,
                    modes=OPERATOR_MODES,
                )
            )
            sources.append(str(sample.source_id))
            tasks.append(str(sample.task_type or ""))
            lengths.append(graph.response_count)
        finally:
            sample.release_attention()

    feature_names = tuple(rows[0])
    for row in rows[1:]:
        if tuple(row) != feature_names:
            raise RuntimeError("mechanism feature schema changed between samples")
    feature = np.asarray(
        [[row[name] for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    frozen_operator.verify(frozen_operator.path)
    directions, probe_groups = frozen_evaluation_spec(feature_names)
    table = FeatureTable(
        sample_id=np.asarray(sample_ids, dtype=str),
        source_id=np.asarray(sources, dtype=str),
        task_type=np.asarray(tasks, dtype=str),
        response_length=np.asarray(lengths, dtype=np.int32),
        feature_names=feature_names,
        feature=feature,
        metadata={
            "labels_used": False,
            "audit_scope": "selected_samples",
            "implementation_sha256": implementation_sha256(),
            "dataset_manifest_sha256": dataset_manifest_sha256(dataset),
            "data_root": str(Path(data_root).resolve()),
            "split": str(dataset.manifest["split"]),
            "task": task,
            "operator_path": str(frozen_operator.path),
            "operator_sha256": frozen_operator.sha256,
            "model_path": geometry.model_path,
            "architecture": geometry.architecture,
            "imputation": imputation,
            "seed": int(seed),
            "layer_count": geometry.layer_count,
            "head_count": geometry.head_count,
            "feature_modes": list(OPERATOR_MODES),
            "feature_directions": directions,
            "probe_groups": probe_groups,
        },
    )
    save_feature_table(output_path, table)
    return {
        "features": str(Path(output_path).resolve()),
        "samples": len(sample_ids),
        "feature_count": len(feature_names),
        "labels_used": False,
        "operator_sha256": table.metadata["operator_sha256"],
    }


def evaluate_features(
    data_root,
    feature_path,
    output_path,
    *,
    bootstrap_replicates: int = 500,
    cv_folds: int = 5,
    seed: int = 20260828,
) -> dict[str, object]:
    """Open answer labels only after the feature artifact has been frozen."""

    if int(bootstrap_replicates) < 0:
        raise ValueError("bootstrap_replicates must be non-negative")

    frozen = FrozenFile.capture(feature_path)
    table = load_feature_table(frozen.path)
    frozen.verify(frozen.path)
    if Path(str(table.metadata["data_root"])).resolve() != Path(data_root).resolve():
        raise ValueError("evaluation data root differs from the frozen feature artifact")
    directions, probe_groups = validate_frozen_evaluation_spec(
        table.feature_names,
        table.metadata["feature_directions"],
        table.metadata["probe_groups"],
    )

    # Complete canonical binding on a label-sealed instance.  The formal cache
    # physically co-locates y_token with attention, so retaining labels even for
    # this pass would make the firewall ordering ambiguous.
    binding_dataset = open_research_dataset(
        data_root,
        device="cpu",
        verify_hashes=True,
        retain_embedded_labels=False,
    )
    source_id = validate_label_free_bindings(table, binding_dataset)
    frozen.verify(frozen.path)

    # Only a second dataset instance is allowed to retain/open labels.
    dataset = open_research_dataset(
        data_root,
        device="cpu",
        verify_hashes=True,
        retain_embedded_labels=True,
    )
    if dataset_manifest_sha256(dataset) != table.metadata["dataset_manifest_sha256"]:
        raise ValueError("labeled dataset manifest differs from feature artifact")
    if str(dataset.manifest.get("split")) != "test":
        raise ValueError("operator mechanism labels may only be opened on test")
    frozen.verify(frozen.path)
    labels = dataset.prepare_evaluation_labels(table.sample_id.tolist())
    answer_label = []
    positive_fraction = []
    for sample_id in tqdm(
        table.sample_id.tolist(),
        total=len(table.sample_id),
        desc="answer labels",
        unit="answer",
    ):
        sample = dataset[sample_id]
        try:
            token_label = labels.response_labels(sample).detach().cpu().numpy().astype(np.int8)
            answer_label.append(int(token_label.any()))
            positive_fraction.append(float(token_label.mean()) if len(token_label) else 0.0)
        finally:
            sample.release_attention()
    frozen.verify(frozen.path)

    answer_label = np.asarray(answer_label, dtype=np.int8)
    positive_fraction = np.asarray(positive_fraction, dtype=np.float64)
    length_baseline = univariate_report(
        np.log1p(table.response_length.astype(np.float64))[:, None],
        ("log_response_length",),
        answer_label,
        source_id,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 100_000,
        directions={"log_response_length": "high"},
    )[0]
    report = {
        "schema": "attention-operator-answer-mechanism-evaluation",
        "version": 1,
        "labels_read": True,
        "labels_used": True,
        "labels_used_during": "posthoc_answer_level_mechanism_validation",
        "feature_artifact": str(frozen.path),
        "feature_sha256": frozen.sha256,
        "dataset_manifest_sha256": table.metadata["dataset_manifest_sha256"],
        "implementation_sha256": table.metadata["implementation_sha256"],
        "operator_sha256": table.metadata["operator_sha256"],
        "samples": len(answer_label),
        "positive_answers": int(answer_label.sum()),
        "prevalence": float(answer_label.mean()),
        "mean_positive_token_fraction": float(positive_fraction.mean()),
        "answer_label_definition": "1 iff any response token is labeled hallucinated",
        "response_length_baseline": length_baseline,
        "univariate": univariate_report(
            table.feature,
            table.feature_names,
            answer_label,
            source_id,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            directions=directions,
        ),
        "source_grouped_logistic_readability": grouped_probe_report(
            table.feature,
            table.feature_names,
            answer_label,
            source_id,
            folds=cv_folds,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            frozen_groups=probe_groups,
            response_length=table.response_length,
        ),
        "claim_boundary": (
            "Operator-aware features are post-hoc answer-level diagnostics. "
            "The source-grouped logistic probe is supervised readability, not "
            "an unsupervised hallucination detector."
        ),
    }
    frozen.verify(frozen.path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**report, "evaluation": str(output_path.resolve())}
