# 2026 native-graph / propagation follow-up

Read primary methods on 2026-09-13 while the frozen native v3 batch runs. These papers were missing from the earlier 2024–2025-focused architecture comparison. No claimed novelty based only on an internal graph, a learned edge gate, or contiguous-span smoothing remains justified. No implementation/result from these papers has been reproduced here.

## SIRG

Primary: https://arxiv.org/html/2601.03052v1 . Read §§3.2–3.4 and appendices C–E.
AttenLRP produces token attribution vectors. Text is split with line breaks and sentence tools; substantive units are extracted, target-side relevance is averaged and source-side relevance uses a maximum. Edges use top-k or the largest drop in sorted weights. Selected source/history fragments are serialized for a fine-tuned AlignScore classifier; response decisions aggregate fragment labels. Thus adaptive edge selection is present, while span segmentation itself starts from textual splitting. It is supervised discrimination, and attribution conservation is not a guarantee of correct constraint ownership. Relevance perturbation tests concern model sensitivity; they do not supply exact lookback-node truth.

Our inference: separate semantic nodes and computational attribution is well motivated, but our required gain must be about applicability-conditioned native evidence and error-versus-recovery dependencies. A top attribution edge alone cannot distinguish copying another event's duration from using the right event.

## CORTEX

Primary: https://arxiv.org/html/2606.31033v1 . Read §§2–3.4,4.2 and appendices B–E.
A frozen observer processes the same answer with and without reference documents. Final-layer state differences form delta features. A second feature subtracts the attention-weighted preceding-answer deltas from each current delta. A supervised MLP classifies tokens. A binary chain with constant label-persistence probability supplies forward–backward smoothing. RAGTruth use is QA; HalluRAG token targets are derived from sentence labels. The appendix explicitly notes that smoothing can merge separate error regions. Its observer replay setting is relevant to our available models.

Our inference: paired high-dimensional sensitivity and history-adjusted features are useful comparators; neither measures unique evidence ownership or actual causal mediation. A constant persistence prior cannot satisfy the user's relation-adaptive boundary requirement. We should compare any future graph boundary with the same emission scores plus a constant chain, retain cross-span dependencies, and evaluate recovery/new-topic boundaries separately. Directly copying delta plus a chain is prior art, not a new method.

## CausalGaze

Primary: https://arxiv.org/html/2604.11087v1 . Read §§3.1–3.3 and A.4/B.5.
The extracted token graph uses hidden states and attention. Edge sensitivity is the absolute product of attention and detector-loss gradient. A learned gate refines edges for a residual graph detector; projected features pass through two graph layers and pooling. Classification uses labeled cross-entropy with a sensitivity penalty. Node saliency also differentiates detector loss. This describes modifications in the detector's extracted graph, not a demonstrated finite intervention on the original LLM's complete continuation.

Our inference: a learned graph edge selector could scale candidate proposals, but detector saliency is a different target from the LLM's answer decision. Calling either 'causal' does not resolve that target mismatch. Our native-event and raw-input-mediated controls remain necessary for the stronger mechanism claim; efficiency measurements must include observer extraction and reruns, not just milliseconds for a small graph classifier.

## Consequence for this project (our design reasoning)

The current audit is a mechanism teacher with strict coverage limits, not an efficient deployed detector. Its semantic entrance has failed in v1/v2. V3 is a frozen test of the observed interface bottleneck; if it still cannot supply real A/B decisions, that is not evidence that internal information is insufficient.

A scalable complete method will require two separately evaluated properties: (1) all-flow scores/boundary proposals from observed high-dimensional states and dependency structure; (2) selective mechanism conclusions only where applicable constraints, native alternatives, matched controls and input-origin mediation are established. Weak reader targets cannot be called independent truth or pure unsupervised labels. The user prefers unsupervised construction; any learned downstream detector must expose exactly which labels or pseudo-labels enter training.

Before adding a trained graph component, resolve supply and supervision: whether valid source events can be aligned without assuming all non-target roles are correct, whether native witnesses cover both wrong continuations and supported recoveries, and whether graph structure improves over identical features without graph edges. Do not treat a conservative all-abstention audit or a smoothed entropy curve as completion.
