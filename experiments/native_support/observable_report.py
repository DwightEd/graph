"""Numeric operator/transition tables and compact per-answer plots."""

import numpy as np
from state_audit.storage import read_arrays, write_csv, write_json


def write_diagnostics(destination, settings):
    operator_rows, transitions = [], []
    reconstruction_error = 0.
    for index, response in enumerate(settings["responses"]):
        directory = destination / "responses" / f"{index:04d}"
        scores = read_arrays(directory / "scores.npz")
        for target in range(len(scores["token_id"])):
            saved = read_arrays(directory / f"token_{target:06d}.npz")
            reconstruction_error = max(reconstruction_error, float(saved["head_reconstruction_error"].max()))
            for layer in range(len(saved["pullback_energy"])):
                spectrum = np.linalg.eigvalsh(saved["pullback_gram"][layer].astype(float)).clip(0)
                mass = spectrum.sum()
                effective_rank = None
                if mass > 0:
                    probability = spectrum[spectrum > 0] / mass
                    effective_rank = float(np.exp(-(probability * np.log(probability)).sum()))
                row = dict(response_id=response["id"], target=target, layer=layer,
                           sketch_effective_rank=effective_rank)
                row.update({name: float(saved[name][layer]) for name in (
                    "pullback_energy", "skip_energy", "ffn_pullback_energy", "pullback_alignment")})
                operator_rows.append(row)
            row = dict(response_id=response["id"], target=target)
            row.update({name: float(scores[name][target]) for name in (
                "source_read_mass", "source_concentration", "response_change", "read_change",
                "source_mass_change", "source_concentration_change", "entropy", "transport_conditional")})
            transitions.append(row)
        plot_answer(directory, response, scores)
    write_csv(destination / "operators.csv", operator_rows, list(operator_rows[0]))
    write_csv(destination / "transitions.csv", transitions, list(transitions[0]))
    write_json(destination / "coverage.json", dict(planned_tokens=sum(
        len(response["token_ids"]) - response["prompt_length"] for response in settings["responses"]),
        scored_tokens=len(transitions), missing_tokens=0, max_head_reconstruction_error=reconstruction_error,
        first_transition="unassessed_NaN_not_zero", spectral_scope="sketch_not_full_J_rank",
        labels_used_for_diagnostics=False))


def plot_answer(directory, response, scores):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for name in ("transport_conditional", "transport_context"):
        axes[0].plot(scores["target"], scores[name], label=name)
    axes[0].set_ylabel("Kernel anomaly")
    for name in ("source_read_mass", "source_concentration"):
        axes[1].plot(scores["target"], scores[name], label=name)
    axes[1].set_ylabel("Source reading")
    for name in ("read_change", "response_change"):
        axes[2].plot(scores["target"], scores[name], label=name)
    axes[2].set_ylabel("State change")
    axes[2].set_xlabel("Answer token index")
    for axis in axes:
        axis.legend(loc="upper right")
    figure.suptitle(f"Answer {response['id']}: observed states, not semantic phase labels")
    figure.savefig(directory / "trajectory.png", dpi=130)
    plt.close(figure)
