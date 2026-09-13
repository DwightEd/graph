# RAGTruth full-population evaluation review — 2026-09-13

## Completion and integrity

`evaluation_1789273472582766558.json` was published before `COMPLETE`; the marker says `completed=17790 failed=0 total=17790`. PID 167241 exited and GPU memory fell to 1 MiB before this review began. The run covered all 17,790 rostered responses with zero failures and 36 task × generator × split groups.

The frozen run settings have `labels_used=false`. The evaluator declares `labels_used_stage=evaluation_only`, checks every completed response against the annotation response/source/generator/split/text hash, and requires complete-response coverage. The official annotation file hash matches the frozen `response.jsonl` hash; the frozen input roster hash also matches. All eight current live files and their execution snapshots match the settings hashes.

This review used the completed evaluation only, plus hashes and frozen code metadata. It did not inspect raw annotations or response text. No external experiment-audit reviewer MCP was available, so the review backend is this independent read-only Codex agent rather than a cross-model integrity verdict.

## Full denominators

- All tokens: 2,903,263; annotated error tokens: 156,478 (5.39%).
- Through first annotated error, inclusive: 2,092,513; error tokens: 7,664 (0.37%).
- Arithmetic post-first complement only: 810,750 tokens and 148,814 errors (18.36%). The evaluator did **not** emit post-first metrics, so no post-first signal comparison is claimed.

The token-weighted observer saved-token mismatch is 24.17%; group rates range from 17.30% to 32.18%. These are observer-replay measurements, not native-generator explanations.

## Descriptive signals across all 36 groups

Token-weighted averages below are summaries of per-group descriptive AUROCs, not a re-pooled AUROC. `error−normal` in the table is the evaluator’s source-balanced conditional mean difference; positive means are larger on annotated error tokens. Each exact mean and its contributing normal/error source counts are in the machine JSON.

| Subset | Signal | Weighted group-AUROC | Group range |
| --- | --- | ---: | ---: |
| all tokens | entropy | 0.577 | 0.460–0.644 |
| all tokens | source JS | 0.446 | 0.353–0.658 |
| all tokens | small-source JS | 0.564 | 0.475–0.608 |
| all tokens | history JS | 0.577 | 0.421–0.729 |
| all tokens | small-history JS | 0.594 | 0.480–0.704 |
| through first error | entropy | 0.799 | 0.645–0.993 |
| through first error | source JS | 0.623 | 0.404–0.769 |
| through first error | small-source JS | 0.739 | 0.575–0.916 |
| through first error | history JS | 0.449 | 0.189–0.953 |
| through first error | small-history JS | 0.457 | 0.213–0.829 |

On all tokens, entropy (0.577), history JS (0.577), and small-history JS (0.594) are only modest descriptive rankings; source JS is below chance in the token-weighted group summary (0.446). Through the first error, entropy (0.799) and small-source JS (0.739) rise, while history JS falls to 0.449. This shift is descriptive, highly class-imbalanced (7,664 first-prefix error tokens), and cannot establish a mechanism.

## Every task / generator / split

Cell format is `AUROC / error−normal source-balanced conditional mean`. `all` is all-token; `first` is through-first-error inclusive. This table includes every frozen group; it does not select the strongest result.

| Task | Generator | Split | Responses | All T/E | First T/E | Mismatch | All entropy | All source | All history | First entropy | First source | First history |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Data2txt | gpt-3.5-turbo-0613 | test | 150 | 40335/502 | 34775/37 | 0.252 | 0.460/0.0208 | 0.419/-0.0383 | 0.421/-0.0265 | 0.645/0.4484 | 0.606/0.0114 | 0.625/0.0631 |
| Data2txt | gpt-3.5-turbo-0613 | train | 883 | 236468/2848 | 202815/235 | 0.253 | 0.511/0.1660 | 0.413/-0.0365 | 0.501/-0.0005 | 0.722/0.6885 | 0.604/0.0218 | 0.678/0.0827 |
| Data2txt | gpt-4-0613 | test | 150 | 25507/216 | 22428/35 | 0.322 | 0.581/0.3053 | 0.485/0.0174 | 0.570/0.0127 | 0.691/0.8880 | 0.534/0.0246 | 0.421/0.0550 |
| Data2txt | gpt-4-0613 | train | 883 | 149617/1717 | 128182/255 | 0.316 | 0.569/0.3286 | 0.494/0.0252 | 0.564/0.0078 | 0.737/0.8399 | 0.648/0.0599 | 0.697/0.0549 |
| Data2txt | llama-2-13b-chat | test | 150 | 33221/2173 | 14774/138 | 0.265 | 0.576/0.2801 | 0.534/0.0217 | 0.551/-0.0026 | 0.709/0.6581 | 0.620/0.0541 | 0.792/0.0779 |
| Data2txt | llama-2-13b-chat | train | 883 | 198223/15647 | 74481/845 | 0.265 | 0.566/0.2549 | 0.523/0.0279 | 0.552/0.0081 | 0.755/0.7159 | 0.644/0.0617 | 0.810/0.0973 |
| Data2txt | llama-2-70b-chat | test | 150 | 28529/1289 | 15227/112 | 0.262 | 0.581/0.2744 | 0.495/0.0134 | 0.569/0.0135 | 0.724/0.6948 | 0.595/0.0190 | 0.653/0.0842 |
| Data2txt | llama-2-70b-chat | train | 883 | 171496/10449 | 74033/751 | 0.266 | 0.573/0.2513 | 0.493/0.0163 | 0.547/0.0066 | 0.748/0.6766 | 0.615/0.0226 | 0.485/0.0719 |
| Data2txt | llama-2-7b-chat | test | 150 | 24941/1473 | 14179/123 | 0.282 | 0.578/0.3253 | 0.523/0.0322 | 0.572/0.0138 | 0.809/0.7677 | 0.684/0.0447 | 0.405/0.0763 |
| Data2txt | llama-2-7b-chat | train | 883 | 149709/10197 | 79359/765 | 0.285 | 0.584/0.3042 | 0.523/0.0241 | 0.587/0.0146 | 0.739/0.7275 | 0.663/0.0540 | 0.557/0.0877 |
| Data2txt | mistral-7B-instruct | test | 150 | 25810/1754 | 10204/134 | 0.312 | 0.581/0.4108 | 0.501/0.0260 | 0.521/-0.0043 | 0.778/0.9041 | 0.572/0.0266 | 0.795/0.0748 |
| Data2txt | mistral-7B-instruct | train | 883 | 154692/12336 | 55751/824 | 0.311 | 0.562/0.2738 | 0.486/0.0035 | 0.526/-0.0030 | 0.746/0.7974 | 0.533/-0.0033 | 0.770/0.0756 |
| QA | gpt-3.5-turbo-0613 | test | 150 | 12415/142 | 12150/5 | 0.179 | 0.625/0.3952 | 0.395/-0.0799 | 0.729/0.0718 | 0.857/1.2414 | 0.404/-0.0946 | 0.498/0.2259 |
| QA | gpt-3.5-turbo-0613 | train | 839 | 75307/1469 | 71182/70 | 0.173 | 0.626/0.2658 | 0.418/-0.0387 | 0.599/0.0366 | 0.889/1.0803 | 0.619/0.0204 | 0.278/0.1181 |
| QA | gpt-4-0613 | test | 150 | 17839/19 | 17821/1 | 0.185 | 0.601/0.2098 | 0.658/0.0828 | 0.632/-0.0091 | 0.993/2.8290 | 0.769/0.1435 | 0.953/0.2005 |
| QA | gpt-4-0613 | train | 839 | 99950/489 | 97041/41 | 0.179 | 0.634/0.2538 | 0.430/-0.0232 | 0.592/0.0159 | 0.834/0.8659 | 0.643/0.0599 | 0.273/0.0580 |
| QA | llama-2-13b-chat | test | 150 | 25117/1559 | 21496/36 | 0.185 | 0.587/0.1972 | 0.366/-0.0728 | 0.636/0.0411 | 0.944/1.1971 | 0.599/-0.0088 | 0.270/0.2114 |
| QA | llama-2-13b-chat | train | 839 | 150831/14979 | 113131/363 | 0.186 | 0.583/0.2101 | 0.391/-0.0483 | 0.624/0.0284 | 0.826/0.8875 | 0.578/0.0137 | 0.420/0.1827 |
| QA | llama-2-70b-chat | test | 150 | 22442/1617 | 19068/35 | 0.182 | 0.583/0.3263 | 0.353/-0.0527 | 0.681/0.0378 | 0.937/1.5097 | 0.612/0.0125 | 0.321/0.2036 |
| QA | llama-2-70b-chat | train | 839 | 139858/12846 | 110908/285 | 0.179 | 0.581/0.1775 | 0.382/-0.0567 | 0.634/0.0397 | 0.906/0.9150 | 0.639/0.0175 | 0.312/0.1792 |
| QA | llama-2-7b-chat | test | 150 | 30721/2307 | 24175/52 | 0.205 | 0.600/0.2807 | 0.417/-0.0407 | 0.606/0.0190 | 0.899/0.8867 | 0.619/0.0258 | 0.306/0.1376 |
| QA | llama-2-7b-chat | train | 839 | 192020/23562 | 126107/458 | 0.197 | 0.575/0.2246 | 0.394/-0.0421 | 0.618/0.0332 | 0.804/0.8232 | 0.595/0.0085 | 0.487/0.1947 |
| QA | mistral-7B-instruct | test | 150 | 16283/1107 | 13922/31 | 0.208 | 0.632/0.4138 | 0.380/-0.0610 | 0.698/0.0444 | 0.951/1.2974 | 0.495/-0.0154 | 0.252/0.1377 |
| QA | mistral-7B-instruct | train | 839 | 102098/11761 | 75903/347 | 0.216 | 0.630/0.3726 | 0.397/-0.0540 | 0.662/0.0346 | 0.874/1.1230 | 0.627/0.0205 | 0.277/0.1591 |
| Summary | gpt-3.5-turbo-0613 | test | 150 | 25017/178 | 24746/4 | 0.210 | 0.552/0.3891 | 0.427/-0.0065 | 0.661/0.0114 | 0.782/0.8989 | 0.515/-0.0333 | 0.652/0.0390 |
| Summary | gpt-3.5-turbo-0613 | train | 793 | 137139/572 | 132150/50 | 0.211 | 0.549/0.1965 | 0.442/-0.0272 | 0.506/0.0054 | 0.771/0.8023 | 0.600/0.0003 | 0.214/0.0761 |
| Summary | gpt-4-0613 | test | 150 | 20091/129 | 19553/6 | 0.232 | 0.474/0.2391 | 0.436/0.0099 | 0.578/-0.0091 | 0.831/0.7080 | 0.759/0.0237 | 0.189/0.0477 |
| Summary | gpt-4-0613 | train | 793 | 107248/774 | 101385/68 | 0.238 | 0.615/0.2848 | 0.392/-0.0498 | 0.548/-0.0026 | 0.808/0.7640 | 0.675/0.0024 | 0.221/0.0337 |
| Summary | llama-2-13b-chat | test | 150 | 17573/382 | 16088/33 | 0.252 | 0.629/0.3554 | 0.474/-0.0235 | 0.604/0.0178 | 0.891/0.8777 | 0.652/0.0198 | 0.281/0.1011 |
| Summary | llama-2-13b-chat | train | 793 | 96456/3177 | 80925/262 | 0.256 | 0.589/0.2603 | 0.491/-0.0076 | 0.600/0.0164 | 0.801/0.6889 | 0.697/0.0663 | 0.460/0.0993 |
| Summary | llama-2-70b-chat | test | 150 | 19722/531 | 17962/24 | 0.207 | 0.583/0.2499 | 0.391/-0.0647 | 0.634/0.0190 | 0.882/0.7987 | 0.685/-0.0239 | 0.276/0.0817 |
| Summary | llama-2-70b-chat | train | 793 | 106649/2326 | 93630/188 | 0.216 | 0.580/0.2282 | 0.414/-0.0489 | 0.599/0.0161 | 0.782/0.6672 | 0.623/0.0194 | 0.336/0.0959 |
| Summary | llama-2-7b-chat | test | 150 | 18434/814 | 15488/51 | 0.240 | 0.619/0.3178 | 0.447/-0.0428 | 0.609/0.0251 | 0.875/0.7373 | 0.534/-0.0288 | 0.349/0.0942 |
| Summary | llama-2-7b-chat | train | 793 | 101477/5167 | 76175/383 | 0.251 | 0.579/0.2511 | 0.460/-0.0252 | 0.595/0.0132 | 0.824/0.5937 | 0.659/0.0177 | 0.307/0.1038 |
| Summary | mistral-7B-instruct | test | 150 | 20411/1451 | 14604/86 | 0.282 | 0.644/0.4865 | 0.413/-0.0559 | 0.609/0.0182 | 0.768/1.0294 | 0.597/0.0255 | 0.321/0.0924 |
| Summary | mistral-7B-instruct | train | 793 | 109617/8519 | 70695/531 | 0.292 | 0.610/0.3357 | 0.413/-0.0431 | 0.601/0.0159 | 0.818/0.8528 | 0.625/0.0017 | 0.323/0.0932 |

## Audit verdict and claim boundary

**Integrity status: PASS with claim warnings.** Population coverage, annotation timing, response/annotation binding, frozen input, and frozen/live code checks passed. The completed files support reporting these observer-replay descriptive sensitivities and the all-group table.

They do not support a factuality detector, source ownership, source-to-output routing, MLP aggregation, original-generator causality, or causal adoption claim. The evaluator itself labels the scope “observer-replay mechanism sensitivity, not validated ownership detector”; its `scientific_review` field remains `REVIEW_UNAVAILABLE`. No bootstrap confidence intervals were produced, and the observed saved-token mismatch prevents treating the observer’s scores as direct generator behavior.

The large all-versus-first difference is compatible with continuous-error contamination after a first error, but only the counts—not post-first metrics—are available. It therefore motivates a separately frozen post-first analysis; it is not evidence for any proposed explanation.

