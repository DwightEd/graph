# Independent engineering review: O4/O5 analysis

Date: 2026-09-13. Reviewer: delegated engineering reviewer using `code-review-and-quality`. Scope: `reanchor/scripts/analyze_relation.py`, the published `reanchor/results/relation_analysis_20260913/numbers.json` and routing figure, and their published O4/O5 raw inputs. Prior engineering reviews remain unchanged.

**Disposition: no Required findings for the published analysis and its stated limited interpretation. Scientific review remains REVIEW_UNAVAILABLE.** This is an engineering and numerical verification, not independent scientific approval or evidence of a general detector.

Only CPU work was performed, with four-thread environment limits. No GPU was used, no population process was signaled, and no implementation or published data was edited. This report is the only new artifact.

## Required

None for the reviewed outputs. The limited conclusions requested by the parent are supported by the verified measurements, with the intervention and example scope stated below.

## Independent raw-data checks

Recomputed all manifest hashes: 190 O4 files and 175 O5 files passed. The manifest hashes recorded in `numbers.json` match the current completed captures. O5's recorded predecessor manifest matches the verified O4 manifest, including the frozen `candidate_queries.json`.

Independently enumerated the O5 condition names and metadata, checking uniqueness and exact sets rather than counts alone: `query_0` through `query_131`, `layer_0` through `layer_31`, `all_queries`, `without_final`, and `sham`. Each single-query condition uses that one step and all 32 layers; each single-layer condition uses step 131 and that one layer. The joint step sets are exactly 0–131 and 0–130. Nonsham donors are world 1; sham uses world 0. The receiver has prompt length 549, endpoint step 131, and candidates `12 = token 717`, `14 = token 975`.

From all 167 O5 full-vocabulary endpoint vectors, independently recomputed the argmax ID, number of maxima, and logit(14) minus logit(12). All matched the saved records. The baseline is unique native 14 with margin +1.25. A strict flip here means unique native 12 over the entire 128256-token vocabulary, relative to that baseline; a candidate-margin crossing alone is not the definition. No O5 endpoint has an argmax tie.

| Condition family | Independent result |
| --- | --- |
| 132 single-query E conditions | Only step 131 flips; its margin is -0.875 |
| 32 single-layer E conditions at step 131 | None flips; margins range +0.5 to +1.5 |
| All 132 query positions jointly patched with E | Unique 12; margin -1.125 |
| Steps 0–130 jointly patched with E | Unique 12; margin -0.25 |
| Same-world final-query E sham | Unique 14; margin +1.25; endpoint exactly equals baseline |

The entire O5 step-131 endpoint vector exactly equals the saved O4 final E endpoint vector. Published joint/sham summaries exactly match the corresponding raw-run records. All saved preceding-query error records are zero, and the sham reports zero across all 132 rows. Those earlier-row invariance claims were guarded during execution; the capture archives only endpoint logits per condition, so this CPU review does not independently reconstruct every earlier row.

Independently recomputed full-vocabulary JS in float64 for all 167 O5 conditions. The maximum absolute difference from saved JS is 6.84e-8. Joint all-query JS is approximately 0.1492657872 nats, and without-final JS is approximately 0.0619578727 nats. Raw sham equality gives JS exactly zero in this recomputation; its saved approximately 5.23e-10 value is floating-point noise and must not be interpreted as an effect.

For O4, independently checked all 160 patch endpoint vectors and four world endpoints, including native ties and candidate margins. All 132 X scan steps are present and none gives a strict native flip. The four fixed final-query comparisons give X 0/4, E 4/4, XE 4/4, MLP 1/4, and numerical-endpoint swap 3/4 native flips. All four unpatched controlled worlds choose their stipulated expected value with a unique native maximum. The published step-99 grill control numbers also agree with the corresponding raw baseline row.

## Readout and retrieval definitions

Recomputed N/A/G choices from the saved readout arrays, averaging all 32 layers at the final query with candidate order 12 then 14. None of those two-candidate readout comparisons is tied. Counts are N 3/4, A 2/4, G 2/4, matching `numbers.json`. These four cases have constructed stage rules and two query contexts; these counts are not natural hallucination-detection accuracy and do not establish graph improvement.

Independently recomputed source-rise scores from O4 receiver attention: sum the source keys, mean heads and layers, take the positive adjacent-step increment, and set the first increment to zero. The maximum numerical difference from saved scores is 5.96e-8; the stable descending top eight are exactly `[11, 36, 16, 31, 94, 2, 115, 88]`. They are eight unique valid steps and were reused unchanged.

The explicitly defined sufficient set is `{131}`: the single-query E interventions yielding unique native 12. Its intersection with the frozen top eight is empty, so recall is 0/1 = 0.0. This is recall against an intervention-specific exhaustive set on this example, not a universal correct-lookback annotation. The analyzer returns null if that sufficient set is empty, as the protocol requires.

## Interpretation and figure

The measurements support: step 131 is sufficient when E is patched across all 32 layers, but no one layer at that step is sufficient under the tested patch. Since jointly patching only steps 0–130 still flips the endpoint, step 131 is not a necessary **patch location in that joint intervention**. Its unpatched computation still participates in the model's forward pass. This result does not establish that the final query is irrelevant, identify a unique earliest event, or establish a general decomposition of causal information.

The PNG was visually inspected. The left panel correctly shows the X/E single-query margin curves and the eight selected step positions; the E curve alone crosses zero at step 131. The right panel shows the 32 single-layer margins, all positive, with the correct +1.25 baseline. Its data are consistent with the full-vocabulary checks above. The black top-eight markers indicate selected positions at a plotting height of +1.25, not source-rise score magnitudes. The PDF is generated from the same figure.

## Optional maintenance improvements

These do not change the published numbers or require modifying frozen experiment code. For future reuse, the analyzer could define sufficiency directly from the independently recomputed token ID and tie count rather than the saved decoded text, and validate exact condition names/metadata rather than counts alone. It could derive the plotted baseline from the loaded baseline vector instead of the fixed +1.25 constant. This review independently checked all three assumptions for the present capture.

## Reviewed artifact SHA-256

- Analyzer: `be12003379d6ff3e9011597c018c4720621b4683029e37067d901af8372cb4cd`
- `numbers.json`: `45e40e0334a6d4f42dfe3e7e36a7a7eb8bfc23e797b6a6a288cfdb237a3ba865`
- PNG: `2b7440a3f0721047f9a70e6771bfd59b638566f76ff20c0331100901823a6550`
- PDF: `1f6b29c8357f8df5fc575f2b6da80dc6e83cab8ac053c641bc996067fbc9a14a`
