# P7 next-source confirmation roster: fixed128 answers before development results

P5 validation and P6 confirmation both failed>0.8. Their cases/results are now
observed. Neither cohort can serve as untouched confirmation for a new method.
P7 has only begun its fixed2 engineering pilot; no P7 natural labels or efficacy
metrics have been joined when this roster rule is written.

Select64 official-test sources:24 QA,24 Summary,16 Data2txt, exactly two saved
Llama2 generators per source (7b-chat and13b-chat), yielding128 original answers.
Order sources within each task by SHA256 of
`P7-confirm-v1|20260914|source_id|source_text_sha256`.
Exclude all R04 source IDs and exact source texts, and all P6 confirmation source
IDs and exact source texts. Deduplicate exact source texts among selected sources.
No labels, quality, model scores, length, confidence or resource-fit filtering.
Missing generator pairs cause a failed roster build, never sample replacement.

Use only the preexisting label-free population inputs, SHA
be388df51e83f60beafbcb4075bde85da11a40f0941279e715f0f334e28d67bb;
R04 SHA3258b79c7886dc2f6f93ff6c6e9e38c695a773b7dab2ddfc4cba271a3baac38f;
spent P6 SHAbc9fabcad838541e8f8892d436a181746c7b200363b9160b73b50ce2291d5101.
Keep original source/prompt/response, generator identity, observer token IDs/offsets.
No annotation file is accepted by the roster builder; model forwards0.

Increasing to64 sources/128 answers aims to reduce the small-source sampling
variability seen in P5/P6, not to guarantee a pass. This is not globally pristine:
the prior full-population observer project measured these sources, and pretraining
contamination is unknown. The final method must separately freeze before scoring.
All128 answers form one endpoint; no stopping at a favorable subset or dropping
truncated/resource-heavy cases. Primary remains all-original-token AUROC>0.8,
with AP, within-answer localization and source-cluster uncertainty disclosed.

This document authorizes roster preparation only, not model scoring before the
method/engineering gate. P7 pilot/development must finish first. New labels stay
unread until all128 frozen predictions complete; a failed point estimate is kept.
