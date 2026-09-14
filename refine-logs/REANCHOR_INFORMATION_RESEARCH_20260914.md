# Reanchor Information Flow for Token Hallucination Analysis

> Research plan, not an implementation guarantee. The current prototype averages
> heads and uses a full-response event quantile and future-attention diagnostics.
> Those measurements are retrospective, not prefix-causal.

## Problem Anchor

We need a label-free, token-level diagnostic for RAGTruth prompt+response
trajectories that explains whether a response token is produced after a
structural reanchor event: a query shifts from local continuation toward an
older prompt/source token or an earlier generated response token.

The method must not define important tokens by a post-hoc high-throughput
threshold. It must first define reanchor events from local-vs-distant routing,
then test whether hallucination onset is enriched at the event or within a
short causal lag after it.

## Paper-grounded hypotheses

### H1: preplan/reanchor structure

Following *Attention Illuminates LLM Reasoning*, define local attention by a
clipped backward window and global influence by future attention received by a
token. A reanchor event is a transition where local routing decreases while
remote prompt/history routing increases. WAAD-like distance and FAI-like
downstream influence are measurements, not labels.

### H2: information filtering differs before correct and incorrect tokens

For a response token, the model's attention distribution over local history,
remote response history, and prompt/source is a membership-like query. Correct
tokens should, under the hypothesis, show a more concentrated and stable
evidence distribution after reanchor; error tokens may show either diffuse
evidence, a high-confidence source/history collision, or a distribution that
looks like a known evidence state despite lacking the required source support.

This is inspired by *Hallucination is a Consequence of Space-Optimality*, but
the theorem does not imply that every RAG hallucination is a random-fact
membership failure. We use its operational objects only: conditional score
distributions, KL/JS separation, overlap, and a rate-distortion-style
trade-off between retained evidence information and false acceptance.

### H3: temporal localization

Hallucination onset is enriched at a reanchor event or within a small forward
lag. The test is source-balanced and reports event prevalence, onset recall,
lead/lag, and permutation/bootstrap intervals. Failure of H3 is a valid result:
reanchor may organize reasoning without locating hallucination.

## Structural measurements

For response query row `t`, with absolute query `q = prompt_length + t - 1`:

- `local_mass`: attention to response keys in `[q-W+1, q]`;
- `remote_history_mass`: attention to earlier response keys `< q-W+1`;
- `prompt_mass`: attention to prompt keys `< prompt_length`;
- `reanchor_distance`: attention-weighted distance to response keys, clipped at W;
- `reanchor_ratio`: `(prompt_mass + remote_history_mass) / (local_mass + eps)`;
- `future_influence`: attention received by response token `s` from later queries;
- `source_entropy`: entropy of prompt/source attention distribution;
- `evidence_concentration`: normalized Herfindahl concentration of prompt/history
  attention;
- `distribution_shift`: JS divergence between the current evidence distribution
  and the preceding local window's distribution.

The event is a predeclared causal rule: a robust quantile exceedance of
reanchor ratio or positive change in clipped distance, with a cooldown. It does
not use labels, future response length, or answer correctness.

## Information-theoretic analysis

For a frozen structural score `z_t`, estimate empirical distributions on
RAGTruth error/non-error tokens only after scores are frozen:

```text
KL(P_correct || P_error)
JS(P_correct, P_error)
overlap(P_correct, P_error)
```

Use source-balanced histograms or fixed quantile bins. Report conditional
versions for `reanchor_event`, `event + delta`, and `no_event`.

For a score interpreted as membership confidence `x_t`, report the empirical
fact/non-fact log-loss pair `(epsilon_correct, epsilon_error)` and the
rate-distortion proxy:

```text
D_hat = KL(P_correct(x) || P_error(x))
```

This is not a theorem about RAGTruth. It is a diagnostic of whether the score
has separable information distributions. A high D_hat with poor onset recall
means the score separates token classes only after the error is visible, not
that it locates the causal source.

## Non-goals

- no claim that attention is causal or that a reanchor node is a semantic
  evidence owner;
- no hallucination classifier trained on RAGTruth labels;
- no high-flow token definition;
- no threshold selected on the official test labels;
- no replacement of source identity/semantic verification with attention mass.
