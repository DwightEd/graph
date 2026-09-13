# Structured alignment follow-up during frozen v2 execution

Only literature/design notes; no changes to running v2 and no new empirical claim.

## SRLScore — read Sections 3.1–3.3, 4.3

Primary paper: https://arxiv.org/pdf/2305.13309 . HTML v2 was unavailable; the primary PDF v1 was read. The method constructs source and summary tuples with agent, negation, relation, patient, recipient, time and location. Each summary tuple selects the best source tuple under weighted role similarities. Weights redistribute across roles actually stated in the summary; omission alone is not falsity. It compares against all source tuples rather than requiring agent/relation agreement first. Optional coreference expands tuple surface forms; the implementation notes that PropBank argument indices do not have universal semantic meanings. Similarity choices include exact matching, word vectors and ROUGE. These are text-level consistency scores, with no native-message mechanism claim.

Project inference: a same-event role table is necessary, but roles/tuples are established prior work, not our novelty. If an entire event contains several wrong roles, single-hidden-role questions can have false remaining premises and abstain across all masks. A future joint source-event alignment must allow this case without treating the nearest event or a high average over roles as proof of support. Keep a separate event identity/qualifier compatibility gate, preserve unresolved competing events, and report multi-role scope. Active/passive grammatical subject must not silently be equated with semantic agent across source/response. This is a possible follow-up, not an implemented or validated v3.

## FENICE — read Sections 3.1–3.4, 4.1

Primary paper: https://aclanthology.org/2024.findings-acl.841.pdf . Claim extraction precedes NLI alignment. Per-premise scoring uses entailment minus contradiction probability; sentence alignment picks the maximum. Coreference variants augment the selected source sentence. Low-scoring claims additionally consider five-sentence blocks and the entire document; reported granularity and trigger thresholds were selected on validation data. The summary score averages claim scores. The paper also distills a claim extractor into T5.

Project inference: larger evidence context is useful for abstraction and reference resolution, while max-over-more-premises is insufficient to identify native adoption or guarantee the correct event. Our high-risk QA gate should not force a noun-span answer for paraphrastic/derived claims. V2 will explicitly reveal failures of this interface. We should consider a compact claim/role extractor or batched verification only after identifying which current reader interface fails. Do not install the older AllenNLP/FENICE dependency stacks into the running shared environment merely to copy the representation. Current v2 remains frozen and evaluation preserves all abstentions.
