# Post-first population results review — 2026-09-13

## Integrity and completion

The completed CPU output has a valid manifest and `results.status=complete`. It records 504 legacy all/first metric-pair checks before publication, has no model forwards, and contains 36 task × generator × official-split groups.

Independent rehashing found the settings hash and both output artifacts valid; all 6 parent files, all 3 live code files and their executed snapshots, and the frozen annotation byte hash match their settings. The response-manifest receipt contains 17,790 entries. This review did not reopen annotation examples or rerun evaluation.

## Denominators and partition

- Full completed population: 17,790 responses, 2,903,263 tokens, 156,478 annotated error tokens.
- Through first error, inclusive: 2,092,513 tokens / 7,664 errors.
- Strictly post-first: 810,750 tokens / 148,814 errors (18.36% raw token prevalence).
- Error runs: 13,664 starts and 142,814 interior tokens. The analysis verifies both all = first + post partitions.

## Post-first descriptive summary

These values are token-weighted averages of **source-balanced group AUROCs**, not pooled AUROCs. AUPRC remains source-balanced too; the 18.36% raw post-first prevalence is not its baseline. All metrics have fixed higher-is-error signs and bootstrap=0.

| Score | Token-weighted group AUROC | Group AUROC range | Groups with both classes |
| --- | ---: | ---: | ---: |
| entropy | 0.5480 | 0.341–0.639 | 35 |
| negative_margin | 0.5312 | 0.362–0.615 | 35 |
| source_small_js | 0.5533 | 0.349–0.608 | 35 |
| history_small_js | 0.5127 | 0.420–0.619 | 35 |
| source_saved_support | 0.4706 | 0.415–0.544 | 35 |
| history_saved_support | 0.4924 | 0.447–0.554 | 35 |
| history_minus_source_support | 0.5222 | 0.428–0.590 | 35 |
| source_permute_js | 0.4942 | 0.371–0.572 | 35 |
| mlp_small_js | 0.5395 | 0.332–0.626 | 35 |

The requested coarse signals are weak after the first error: entropy 0.5480, source-small JS 0.5533, history-small JS 0.5127, history-minus-source support 0.5222, and MLP-small JS 0.5395. These values answer the missing post-first descriptive question, but do not establish source ownership, routing, continuous-error detection, or causality.

## All 36 groups

The table includes every frozen task/generator/split group. Metrics are post-first AUROC; full nine-score AUROC/AUPRC values are in the machine JSON.

| Task | Generator | Split | Responses | Post T/E/S | Entropy | Source-small | History-small | H−S support | MLP-small |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Data2txt | gpt-3.5-turbo-0613 | test | 150 | 5560/465/37 | 0.463 | 0.473 | 0.450 | 0.483 | 0.445 |
| Data2txt | gpt-3.5-turbo-0613 | train | 883 | 33653/2613/235 | 0.488 | 0.485 | 0.492 | 0.483 | 0.498 |
| Data2txt | gpt-4-0613 | test | 150 | 3079/181/35 | 0.453 | 0.530 | 0.438 | 0.505 | 0.473 |
| Data2txt | gpt-4-0613 | train | 883 | 21435/1462/255 | 0.479 | 0.488 | 0.493 | 0.515 | 0.488 |
| Data2txt | llama-2-13b-chat | test | 150 | 18447/2035/138 | 0.550 | 0.569 | 0.496 | 0.524 | 0.536 |
| Data2txt | llama-2-13b-chat | train | 883 | 123742/14802/845 | 0.540 | 0.566 | 0.494 | 0.520 | 0.525 |
| Data2txt | llama-2-70b-chat | test | 150 | 13302/1177/112 | 0.539 | 0.547 | 0.504 | 0.526 | 0.527 |
| Data2txt | llama-2-70b-chat | train | 883 | 97463/9698/751 | 0.554 | 0.565 | 0.510 | 0.533 | 0.541 |
| Data2txt | llama-2-7b-chat | test | 150 | 10762/1350/123 | 0.558 | 0.591 | 0.522 | 0.487 | 0.556 |
| Data2txt | llama-2-7b-chat | train | 883 | 70350/9432/765 | 0.538 | 0.572 | 0.500 | 0.534 | 0.535 |
| Data2txt | mistral-7B-instruct | test | 150 | 15606/1620/134 | 0.532 | 0.566 | 0.468 | 0.504 | 0.517 |
| Data2txt | mistral-7B-instruct | train | 883 | 98941/11512/824 | 0.530 | 0.566 | 0.495 | 0.525 | 0.526 |
| QA | gpt-3.5-turbo-0613 | test | 150 | 265/137/5 | 0.595 | 0.582 | 0.578 | 0.543 | 0.575 |
| QA | gpt-3.5-turbo-0613 | train | 839 | 4125/1399/70 | 0.602 | 0.570 | 0.500 | 0.528 | 0.575 |
| QA | gpt-4-0613 | test | 150 | 18/18/1 | NA | NA | NA | NA | NA |
| QA | gpt-4-0613 | train | 839 | 2909/448/41 | 0.595 | 0.561 | 0.519 | 0.537 | 0.583 |
| QA | llama-2-13b-chat | test | 150 | 3621/1523/36 | 0.583 | 0.548 | 0.547 | 0.498 | 0.555 |
| QA | llama-2-13b-chat | train | 839 | 37700/14616/363 | 0.565 | 0.540 | 0.526 | 0.505 | 0.553 |
| QA | llama-2-70b-chat | test | 150 | 3374/1582/35 | 0.607 | 0.561 | 0.571 | 0.523 | 0.597 |
| QA | llama-2-70b-chat | train | 839 | 28950/12561/285 | 0.570 | 0.540 | 0.528 | 0.514 | 0.556 |
| QA | llama-2-7b-chat | test | 150 | 6546/2255/52 | 0.584 | 0.563 | 0.541 | 0.502 | 0.567 |
| QA | llama-2-7b-chat | train | 839 | 65913/23104/458 | 0.573 | 0.549 | 0.548 | 0.520 | 0.564 |
| QA | mistral-7B-instruct | test | 150 | 2361/1076/31 | 0.639 | 0.589 | 0.598 | 0.539 | 0.626 |
| QA | mistral-7B-instruct | train | 839 | 26195/11414/347 | 0.591 | 0.564 | 0.557 | 0.532 | 0.588 |
| Summary | gpt-3.5-turbo-0613 | test | 150 | 271/174/4 | 0.570 | 0.608 | 0.619 | 0.428 | 0.574 |
| Summary | gpt-3.5-turbo-0613 | train | 793 | 4989/522/50 | 0.573 | 0.548 | 0.538 | 0.523 | 0.559 |
| Summary | gpt-4-0613 | test | 150 | 538/123/6 | 0.341 | 0.349 | 0.420 | 0.590 | 0.332 |
| Summary | gpt-4-0613 | train | 793 | 5863/706/68 | 0.552 | 0.518 | 0.539 | 0.542 | 0.548 |
| Summary | llama-2-13b-chat | test | 150 | 1485/349/33 | 0.602 | 0.550 | 0.546 | 0.498 | 0.580 |
| Summary | llama-2-13b-chat | train | 793 | 15531/2915/262 | 0.558 | 0.528 | 0.533 | 0.516 | 0.550 |
| Summary | llama-2-70b-chat | test | 150 | 1760/507/24 | 0.591 | 0.568 | 0.577 | 0.514 | 0.588 |
| Summary | llama-2-70b-chat | train | 793 | 13019/2138/188 | 0.552 | 0.528 | 0.518 | 0.538 | 0.535 |
| Summary | llama-2-7b-chat | test | 150 | 2946/763/51 | 0.588 | 0.559 | 0.567 | 0.491 | 0.576 |
| Summary | llama-2-7b-chat | train | 793 | 25302/4784/383 | 0.563 | 0.553 | 0.532 | 0.534 | 0.559 |
| Summary | mistral-7B-instruct | test | 150 | 5807/1365/86 | 0.591 | 0.567 | 0.544 | 0.541 | 0.586 |
| Summary | mistral-7B-instruct | train | 793 | 38922/7988/531 | 0.576 | 0.548 | 0.543 | 0.543 | 0.563 |

## Claim boundary

**Integrity: pass. Scientific claim: descriptive negative/weak result.** The result is reproducibly tied to the frozen completed observer measurements and official annotations joined after those measurements. It supports reporting that these fixed coarse sensitivity scores do not provide a strong, consistent post-first signal. It supplies no confidence intervals, no trained detector, no causal intervention on source owners, no evidence of message routing, and no original-generator attribution.

