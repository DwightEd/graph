# Token-level behavior analysis

This analysis is designed for studying *how* a response transitions into hallucination, rather than only asking whether hallucinated and correct samples separate globally under t-SNE.

## Research design

The analysis keeps graph feature extraction label-free. `positive_runs` is read only after token-level graph behavior features are computed, and is used to locate hallucination spans for visualization and aligned comparisons.

For each response token, `behavior.token_behavior_features` returns 11 features:

| Feature | Meaning | Expected hallucination signature to test |
| --- | --- | --- |
| `incoming_mass` | Sum of mean-channel retained incoming edge weights | overall routing strength changes |
| `prompt_mass_share` | Fraction of incoming mass from prompt tokens | decreases if grounding weakens |
| `normalized_entropy` | Normalized entropy of incoming edge weights | decreases if routing concentrates |
| `history_lag` | Attention-weighted normalized distance to response-history sources | decreases if dependencies become more local |
| `in_degree` | Number of retained incoming edges | decreases if the graph becomes sparser |
| `prompt_degree` | Number of retained prompt-to-response edges | decreases if prompt connectivity weakens |
| `history_degree` | Number of retained response-history edges | measures self-history connectivity |
| `in_density` | `in_degree / number_of_possible_previous_sources` | length-normalized sparsity |
| `prompt_density` | `prompt_degree / prompt_length` | length-normalized prompt connectivity |
| `history_density` | `history_degree / available_response_history` | local self-history connectivity |
| `history_edge_share` | Fraction of incoming edges coming from response history | increases if the response becomes self-dependent |

The original 4-column `token_routing_features` and `graph_tsne.ipynb` are unchanged. The first four columns above are exactly the existing routing features, so old t-SNE results remain comparable.

## 1. Single-sample case study

Use one sample to inspect token-level trajectories, hallucination spans, and the sparse routing map.

```bash
python scripts/analyze_behavior.py single \
  --attention-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b/test \
  --graph-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/llama31_8b/relation_topk_channels/test \
  --sample-id <hallucinated_sample_id> \
  --output-dir outputs/behavior/<hallucinated_sample_id>
```

Outputs:

- `behavior.csv`: one row per response token, including token ID, hallucination flag, and all 11 behavior features.
- `run_summary.csv`: mean feature values in `pre`, `error`, `post`, and `error_minus_pre` windows for every hallucination span.
- `behavior.png`: normalized behavior trajectories, retained edge counts, incoming mass, and sparse source-to-target routing map. Hallucination spans are shaded.
- `metadata.json`: sample ID, source ID, response length, positive runs, and feature names.

A fully correct sample can be overlaid using normalized response position:

```bash
python scripts/analyze_behavior.py single \
  --attention-root /path/to/canonical/test \
  --graph-root /path/to/graphs/test \
  --sample-id <hallucinated_sample_id> \
  --control-sample-id <fully_correct_sample_id> \
  --output-dir outputs/behavior/pair
```

This additionally writes `error_vs_control.png`.

## 2. Hallucination-onset alignment

To test whether a pattern is systematic rather than a single anecdotal case, align many samples at hallucination onset (`relative_position = 0`).

```bash
python scripts/analyze_behavior.py align \
  --attention-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b/test \
  --graph-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/llama31_8b/relation_topk_channels/test \
  --radius 12 \
  --run-policy first \
  --output-dir outputs/behavior/onset_test
```

By default each hallucination event is paired with a fully correct control. Matching first prefers the same `source_id` and the nearest response length. If no correct sample shares that source, it falls back to the globally nearest response length. The control center is placed at the same normalized response position as the hallucination onset.

Outputs:

- `onset_alignment.npz`: raw aligned error and control windows.
- `onset_summary.csv`: mean, population standard deviation, valid count, and error-minus-control difference for every feature and relative token position.
- `onset_alignment.png`: aggregate trajectories for the five main hypotheses (`prompt_mass_share`, `normalized_entropy`, `history_lag`, `in_density`, `history_edge_share`).
- `matched_events.csv`: every error/control pairing and alignment position.
- `metadata.json`: analysis settings and event count.

Use `--run-policy all` to treat every hallucination span as a separate event. Use `--no-controls` for a pure within-error onset analysis. `--max-events N` provides a deterministic small run for debugging.

## Recommended interpretation sequence

1. **Case discovery:** inspect several single hallucinated samples and identify repeatable changes around the labeled span.
2. **Within-sample transition:** use `error_minus_pre` to test whether the response changes state when hallucination begins.
3. **Matched control:** check that the same change is not simply the normal effect of progressing later in a response.
4. **Onset aggregation:** align many events at token 0 and test whether the same transition appears consistently.
5. **Global visualization:** use the existing t-SNE notebook only after the local behavior signature has been characterized.

The main hypotheses can therefore be tested as directional transitions around hallucination onset:

- weaker prompt grounding: `prompt_mass_share ↓`, `prompt_density ↓`;
- stronger self-history reliance: `history_edge_share ↑`, possibly `history_density ↑`;
- sparser routing: `in_degree ↓`, `in_density ↓`;
- more local routing: `history_lag ↓`;
- more concentrated routing: `normalized_entropy ↓`.

These are hypotheses to validate statistically, not assumptions built into the feature extraction.
