"""Command-line entry point for exact frozen operator graph construction."""

from __future__ import annotations

import argparse
import json

from .config import GraphConstructionConfig
from .pipeline import construct_split


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Construct label-free token graphs from exact frozen Transformer "
            "attention/value/residual computations"
        )
    )
    result.add_argument("--split-root", required=True)
    result.add_argument(
        "--source-json",
        required=True,
        help=(
            "Raw RAGTruth response.jsonl used only for immutable source "
            "provenance; graph features are built from the token-aligned formal cache"
        ),
    )
    result.add_argument("--model-path", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--device", required=True)
    result.add_argument(
        "--model-dtype",
        required=True,
        choices=("float32", "float16", "bfloat16"),
        help="Must match the dtype used to create the formal attention cache.",
    )
    result.add_argument("--sample-id", action="append", default=None)
    result.add_argument("--limit", type=int, default=None)
    result.add_argument("--route-mass-retention", type=float, default=1.0)
    result.add_argument("--value-energy-retention", type=float, default=1.0)
    result.add_argument("--minimum-role-edges", type=int, default=1)
    result.add_argument("--conservation-atol", type=float, default=5e-3)
    result.add_argument("--conservation-rtol", type=float, default=5e-3)
    result.add_argument("--cache-binding-atol", type=float, default=5e-3)
    result.add_argument("--feature-epsilon", type=float, default=1e-8)
    result.add_argument(
        "--output-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    result.add_argument("--verify-hashes", action="store_true")
    result.add_argument("--allow-remote-files", action="store_true")
    result.add_argument("--trust-remote-code", action="store_true")
    result.add_argument("--revision", default=None)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    arguments = parser().parse_args()
    config = GraphConstructionConfig(
        route_mass_retention=arguments.route_mass_retention,
        value_energy_retention=arguments.value_energy_retention,
        minimum_role_edges=arguments.minimum_role_edges,
        conservation_atol=arguments.conservation_atol,
        conservation_rtol=arguments.conservation_rtol,
        cache_binding_atol=arguments.cache_binding_atol,
        feature_epsilon=arguments.feature_epsilon,
        output_dtype=arguments.output_dtype,
    )
    report = construct_split(
        split_root=arguments.split_root,
        source_json=arguments.source_json,
        model_path=arguments.model_path,
        output_root=arguments.output_root,
        device=arguments.device,
        model_dtype=arguments.model_dtype,
        config=config,
        sample_ids=arguments.sample_id,
        limit=arguments.limit,
        verify_hashes=arguments.verify_hashes,
        local_files_only=not arguments.allow_remote_files,
        trust_remote_code=arguments.trust_remote_code,
        revision=arguments.revision,
        overwrite=arguments.overwrite,
    )
    print(
        json.dumps(
            {
                "output_root": str(report.output_root),
                "count": int(report.manifest["count"]),
                "index_sha256": report.manifest["index_sha256"],
                "feature_contract_sha256": report.manifest[
                    "feature_contract_sha256"
                ],
                "labels_read_during_construction": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
