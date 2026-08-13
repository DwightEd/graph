# Structure-preserving token graph representation

## Scientific contract

The deterministic statistics in `attention_graph.statistics` are the canonical base state and remain directly recoverable in every saved token artifact. No PCA or learned encoder defines the detector. Evaluation labels remain sealed until representations, scores, sample selection and per-sample graph artifacts have been written.

## Base state and score

The full base state contains the historical pair-graph statistics, absolute mean edge strength and exact direct Lookback. Train-only position-conditioned median/MAD produces standardized values. The primary score uses five pre-registered directions: low prompt mass fraction, low edge density, high concentration, high retained mean edge strength and low RR lag. `top1_share` remains recoverable but is not counted again because it overlaps the concentration mechanism. Missing RR support masks lag. Exact Lookback remains an independently evaluated baseline rather than being diluted by the composite.

\[
z_{td}=\frac{x_{td}-\operatorname{median}_{train,d}}
{\operatorname{MAD}_{train,d}},\qquad
e_{td}=\max(\sigma_dz_{td},0).
\]

The token score is the mean of these one-sided evidence values. This preserves direction, unlike nearest-cluster distance.

This run is exploratory if these directions were chosen after inspecting the
same benchmark test labels. Runtime label sealing prevents implementation
leakage, not researcher-level test-set selection. Confirmatory evaluation must
freeze features, directions and hop depth on an independent validation split.

## Raw typed propagation

The RP/RR pair weight is the sum of retained layer/head traces divided by channel count. Neither relation matrix is row-normalized and censored mass is not redistributed.

With `W=A_RR`, both raw and conditional quantities are stored:

\[
M^{(k)}=W^kZ,\qquad q^{(k)}=W^k\mathbf1,
\]

\[
\bar Z^{(k)}=M^{(k)}/q^{(k)},\qquad
\Delta^{(k)}=Z-\bar Z^{(k)}.
\]

Prompt provenance is

\[
p^{(0)}=A_{RP}\mathbf1,\qquad p^{(k)}=W^kp^{(0)}.
\]

The saved vector concatenates standardized exact features, each hop's raw message, conditional ancestor mean, self/ancestor residual, RR/RP path mass and reachable ancestor count. Default depth is two.

A hop is causally eligible for response position (t) only when (t\ge k). Structurally impossible early hops are excluded, while an eligible but missing RP path remains weak-path evidence. Conditional innovations require an actual retained path and are continuously gated by (q/(q+q_0)), where (q_0) is the frozen train-positive path-mass median for that position bin and hop. Thus reliability is zero at zero mass, tends continuously to zero for numerical traces, and approaches one only for strong paths. RR path deficit is reported diagnostically but is not part of the primary score because its anomaly direction has not been independently validated.

## Frozen ablations

`token_only`, `token_graph`, `no_rp` and `no_rr` share the same train-only scalers and scoring formula. `no_rp` and `no_rr` really delete the corresponding edge type and recompute the exact node statistics before scoring; no view refits a projector or detector. Claims are evaluated through `token_graph-token_only`, `token_graph-no_rp` and `token_graph-no_rr` differences.

The primary per-sample graph places every token at `(fixed token mechanism evidence, multi-hop graph evidence)` and overlays RR edges. PCA is fitted on a bounded train-only reference and is retained only as a population-level auxiliary diagnostic. Neither coordinate system affects scores.

## Metric interpretation

Every exact feature reports signed raw AUROC and `separability=max(AUC,1-AUC)` separately. Separability is retrospective either-direction association, not deployable signed AUROC. Token and response-aggregated results are both reported so different granularities are not conflated.
