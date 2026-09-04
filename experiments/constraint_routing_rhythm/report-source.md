# Research source: constraint routing rhythm

This note records the evidence used to freeze the method before looking at
RAGTruth labels.  It is not an experimental result.

## Positive design pattern

**Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm Enables
Fine-grained Policy Optimization** (arXiv:2510.13554v2, ICML 2026) separates
heads by average attention distance, visualizes a local sawtooth rhythm and
future global anchors, defines WAAD and FAI from those observations, perturbs
high-FAI tokens, and only then uses the signals for RL credit.  The paper also
states that attention is a filtered signal rather than a complete causal
decomposition.  This motivates our ordering:

1. visualize a functional local/global rhythm;
2. freeze a compact route proposal rule without labels;
3. verify output control with a real message deletion;
4. evaluate one preregistered causal score.

Primary sources:

- https://arxiv.org/html/2510.13554v2
- https://openreview.net/forum?id=ld40RIJnpb

## Novelty boundaries

**How Does Reasoning Flow? Tracing Attention-Induced Information Flow for
Targeted RL in LLMs** (arXiv:2606.10646) already constructs an
attention-induced token DAG, reweights it toward the answer, and extracts
multi-hop flow backbones.  Therefore a token attention graph, path product, or
flow statistic alone is not our contribution.

**HAVE: Head-Adaptive Gating and Valuation for Hallucination Mitigation in
Large Language Models** (arXiv:2509.06596) already combines attention with
Value magnitude for head gating.  Therefore replacing attention by `A*||V||`
alone is not our contribution.

**Attention Sinks as Early Warning Signals for Hallucinations in Large
Language Models** (arXiv:2604.10697) uses future attention to token sinks and a
supervised probe, and explicitly treats the result as associative rather than
causal.  Therefore future influence is only an explanatory route proposal in
our method, never the detector.

Primary sources:

- https://arxiv.org/html/2606.10646v1
- https://arxiv.org/html/2509.06596v1
- https://arxiv.org/html/2604.10697v2

## Frozen method decision

The descriptive map is the exact per-head residual-write magnitude

\[
\kappa_{l,h,q,s}=A_{l,h,q,s}\,\|W^O_{l,h}V_{l,h,s}\|_2.
\]

It supports two label-free descriptive quantities: a windowed functional
backward reach and an ordered evidence-to-carrier-to-future bottleneck.  The
latter is the minimum of early evidence binding and late future influence,
with a separate absolute-message floor.  These quantities visualize route
availability and propose carrier endpoints; neither is called causal.

The only primary detection score deletes all post-softmax Value messages whose
source is in the declared evidence span, without renormalizing attention, then
reruns every downstream Transformer operation.  With the baseline target and
runner fixed before intervention,

\[
C_t = r_t^{-E}-r_t^0.
\]

The score is deliberately not divided by the baseline margin. Higher `C_t`
means removing evidence did not hurt, or helped, the model's recorded token.
This is an output-sensitivity deficit to the evidence Value channel. It is not
proof of incorrectness: parametric knowledge, misleading evidence, and Q/K or
MLP pathways are explicit counterexamples.

On a fixed label-blind audit subset only, an upstream evidence-to-carrier cut,
a later carrier-to-response cut, and their union test non-additivity.  This
diagnostic can support an evidence-conditioned relay interpretation but cannot
establish semantic mediation by itself.

## Stopped direction

The learned graph surrogate and approximate minimum-sufficient-circuit
frontier are removed from the main method.  They require many reruns, introduce
approximation and search error, and overlap established circuit-discovery
work.  They may be revisited only if the lean causal score first establishes a
repeatable empirical phenomenon.
