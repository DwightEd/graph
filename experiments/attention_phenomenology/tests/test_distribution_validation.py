import json
from types import SimpleNamespace

import numpy as np
import torch

from experiments.attention_phenomenology.config import PhenomenologyConfig
from experiments.attention_phenomenology.distribution_validation import (
    DistributionValidationConfig,
    PriorityReservoir,
    validate_composition_distributions,
)


def test_priority_reservoir_is_bounded_and_tracks_population():
    reservoir = PriorityReservoir(16, np.random.default_rng(4))
    reservoir.add_batch(np.arange(200, dtype=np.float32).reshape(100, 2))

    assert reservoir.seen == 100
    assert reservoir.matrix().shape == (16, 2)


class Sample:
    def __init__(self, sample_id, analysis):
        self.sample_id = sample_id
        self.task_type = "QA"
        self.analysis = analysis

    def release_attention(self):
        pass


class Dataset:
    def __init__(self, samples):
        self.samples = {sample.sample_id: sample for sample in samples}
        self.sample_ids = list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[sample_id]


def make_analysis(seed):
    rng = np.random.default_rng(seed)
    values = rng.dirichlet([3, 2, 1, 2], size=4 * 2 * 3)
    role = torch.tensor(values.reshape(4, 2, 3, 4), dtype=torch.float32)
    prompt, response, self_mass, unresolved = role.unbind(-1)
    routing = SimpleNamespace(
        role_probability=role,
        prompt_mass=prompt,
        response_mass=response,
        self_mass=self_mass,
        unresolved_mass=unresolved,
    )
    aggregate = torch.zeros((4, 3))
    head_lower = prompt.clone()
    unsupported = response * 0.25
    provenance = SimpleNamespace(
        aggregate_lower=aggregate,
        head_lower=head_lower,
        unsupported_response_lower=unsupported,
    )
    return SimpleNamespace(
        routing=routing,
        provenance=provenance,
        layer_features=torch.zeros((4, 2, 1)),
    )


def test_validation_writes_label_free_model_comparison(tmp_path, monkeypatch):
    fit = Dataset([Sample(str(i), make_analysis(i)) for i in range(12)])
    validation = Dataset([Sample(str(i), make_analysis(100+i)) for i in range(12)])

    def open_dataset(path, device="cpu"):
        return fit if str(path) == "fit" else validation

    monkeypatch.setattr(
        "experiments.attention_phenomenology.distribution_validation.open_research_dataset",
        open_dataset,
    )
    monkeypatch.setattr(
        "experiments.attention_phenomenology.distribution_validation.collect_routing_edges",
        lambda sample, config=None: sample.analysis,
    )
    monkeypatch.setattr(
        "experiments.attention_phenomenology.distribution_validation.analyze_routing",
        lambda analysis, config=None: analysis,
    )
    summary = validate_composition_distributions(
        fit_split="fit",
        validation_split="validation",
        output_dir=tmp_path,
        phenomenology_config=PhenomenologyConfig(causal_position_bins=1),
        validation_config=DistributionValidationConfig(
            fit_reservoir_rows=128,
            validation_reservoir_rows=128,
            minimum_group_rows=16,
            pseudocounts=(1e-4,),
            simulation_rows=128,
        ),
    )

    assert summary["labels_read"] is False
    assert summary["evaluated_groups"] == 4
    assert (tmp_path / "group_metrics.csv").exists()
    reference = json.loads((tmp_path / "reference.json").read_text())
    assert reference["labels_read"] is False
