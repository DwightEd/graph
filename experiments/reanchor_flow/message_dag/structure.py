"""Graph-derived observables. Names describe measurements, not mechanisms."""
import numpy as np

NAMES = ("material_output_signed_share", "material_relay_share", "local_relay_share",
         "carrier_concentration", "relay_cancellation", "material_unit_concentration")
HEAD_NAMES = ("material_boundary", "material_relay", "response_initial_relay")
SCORES = ("material_deficit", "direct_deficit", "negative_logprob", "position")


def ratio(a,b):
    return float(a/b) if b > 0 else np.nan


def describe(graph, trace):
    material = graph["source_kinds"] == "material"
    history = graph["source_kinds"] == "response_initial"
    output = graph["source_output"]
    total = np.abs(output).sum()
    relay = graph["head_relay_absolute"]
    carrier = graph["carrier_absolute"].sum((0,1))
    mass = np.abs(output[material])
    values = [ratio(output[material].sum(), total) if material.any() else np.nan,
              ratio(relay[material].sum(), relay.sum()) if material.any() else np.nan,
              ratio(graph["local_relay_absolute"].sum(), relay.sum()),
              ratio(np.square(carrier).sum(), carrier.sum()**2),
              1-ratio(abs(graph["head_relay"].sum()), relay.sum()),
              ratio(np.square(mass).sum(), mass.sum()**2)]
    q = int(graph["query"] - trace["row_position"][0])
    direct = graph["boundary_at_target"][material].sum()
    material_total = output[material].sum()
    scores = [-ratio(material_total,abs(material_total)+abs(output.sum()-material_total)) if material.any() else np.nan,
              -ratio(direct, abs(direct)+abs(output.sum()-direct)) if material.any() else np.nan,
              -float(trace["predictor_logprob"][q]), float(graph["target"]-trace["response_start"])]
    heads = np.stack((graph["head_boundary"][material].sum(0), graph["head_relay"][material].sum(0),
                      graph["head_relay"][history].sum(0)))
    if not material.any(): heads[:2] = np.nan
    return {"metric_names": np.array(NAMES), "metrics": np.array(values),
            "head_metric_names": np.array(HEAD_NAMES), "head_metrics": heads,
            "score_names": np.array(SCORES), "scores": np.array(scores)}
