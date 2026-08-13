# Method: mechanism-guided token representation with graph ablations

## 1. Layer-head mechanism tensor

For response token (t), layer (l), and head (h), let (P) be retained prompt attention, (R) retained response-history attention, and (D) the saved diagonal. The method computes four non-redundant channels without averaging layers or heads.

Routing balance:

\[
b_{lht}=\frac{P/N_p}{P/N_p+(R+D)/(t+1)}.
\]

Effective support fraction uses the Herfindahl concentration of all retained strict edges and the diagonal:

\[
q_{lht}=\frac{1}{(N_p+t+1)\sum_j (a_j/(P+R+D))^2}.
\]

Dominant strength is \(\max_j a_j\). Response locality is the RR/diagonal-weighted normalized inverse lag, with the diagonal assigned locality one. The resulting raw node tensor is

\[
X_t\in\mathbb R^{L\times H\times 4}.
\]

Unretained mass caused by the cache floor is saved as a control and is not redistributed or included in (X_t).

## 2. Train-only token embedding

The tensor is flattened only after preserving its layer/head coordinates. A bounded random sample of train tokens fits coordinate-wise median/MAD scaling followed by PCA whitening. No hallucination label, test token, layer ranking, or supervised probe is used. This produces the direct token representation (z_t).

## 3. Fixed typed graph propagation

The canonical CSR support is converted into its full retained RP/RR pair graph. Exact source positions receive deterministic sinusoidal codes. For every response target:

- RP aggregates normalized prompt-source codes into direct provenance (B^{(0)});
- (H^{(k)}=P_{RR}H^{(k-1)}) transports token state over exact (k)-edge RR paths;
- (B^{(k)}=P_{RR}B^{(k-1)}) transports prompt provenance through response intermediates;
- source-position states follow the same recurrence, and each node stores hop-wise reachable-ancestor count and influence mass.

The concatenation is projected by a second train-only robust PCA. There are no trainable graph weights and no backpropagation. Three graph views use the same protocol: full `token_graph`, `no_rp`, and `no_rr`. `token_only` is the direct mechanism embedding.

## 4. Unsupervised score and evaluation

Each view fits MiniBatch K-Means prototypes on bounded train tokens. A test token score is its nearest-prototype distance divided by the train median distance within that prototype. The anomaly direction is fixed before labels are opened.

All test embeddings, scores, sample selection, and per-sample graph artifacts are persisted label-free. Only then is the evaluation dataset opened to compute AUROC/AUPRC and color plots. The main scientific comparison is `token_graph - token_only`; `no_rp` and `no_rr` identify which relation supplies any gain.

PCA coordinates are visual diagnostics, not the detector and not proof of separability. Every sample has a saved graph artifact. Requested sample figures have three panels: retained direct RP/RR edges; non-adjacent effective RR and inherited RP relations computed from (P_{RR}^2,P_{RR}^3); and every response token in the frozen graph-embedding coordinates.
