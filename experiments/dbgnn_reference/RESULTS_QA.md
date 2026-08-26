# QA result: first-order GCN versus order-2 DBGNN

## Setup

```text
samples           149
response tokens   30,619
positive tokens    2,307
prevalence          7.53%
embedding dim          64
```

All encoders were trained without hallucination labels using endpoint prediction. The downstream unsupervised readers and supervised readability probes used only the frozen token embeddings.

## Main result

| Representation | Reader | AUROC | AUPRC |
|---|---|---:|---:|
| GCN | PCA-kNN | 0.6982 | 0.1617 |
| GCN | Isolation Forest | 0.6362 | 0.1411 |
| GCN | Autoencoder | 0.5649 | 0.0935 |
| GCN | linear supervised probe | 0.7865 | 0.2999 |
| GCN | MLP supervised probe | 0.7785 | 0.2760 |
| causal DBGNN | every unsupervised reader | 0.5000 | 0.0753 |
| causal DBGNN | linear supervised probe | 0.5127 | 0.0786 |
| causal DBGNN | MLP supervised probe | 0.4974 | 0.0760 |
| no-transition DBGNN | every unsupervised reader | 0.5000 | 0.0753 |
| no-transition DBGNN | linear supervised probe | 0.5144 | 0.0787 |
| position | PCA-kNN | 0.5234 | 0.0821 |
| position | linear supervised probe | 0.6087 | 0.1045 |

## Interpretation

The first-order GCN representation is clearly useful. Its PCA-kNN score improves over the position PCA-kNN baseline by `+0.1748` AUROC and `+0.0796` AUPRC. The source-disjoint linear probe reaches `0.7865 / 0.2999`, so the frozen GCN embedding contains substantially more correctness information than the current unsupervised detector extracts.

The current order-2 DBGNN adaptation fails. Both the causal-transition and no-transition variants produce effectively collapsed calibration geometry: every unsupervised reader returns a constant score, so AUROC is `0.5` and AUPRC equals prevalence. Their supervised probes are also approximately random. The exact zero unsupervised delta between causal and no-transition is therefore not evidence that the two graph constructions are equivalent; it is a consequence of both representations being unusable to the registered readers.

The causal transition adds no measurable readability. The causal-minus-no-transition linear-probe delta is `-0.0017` AUROC with a bootstrap interval spanning zero, and the MLP delta is exactly zero.

## What this experiment supports

1. A simple first-order attention graph can produce useful token embeddings under label-free endpoint prediction.
2. The detector is part of the remaining bottleneck for GCN, because the supervised linear ceiling is much higher than PCA-kNN.
3. The current De Bruijn lifting and HO-GCN adaptation should not be used as the main method.
4. More graph complexity is not automatically beneficial; the order-2 variants are substantially worse than the simpler GCN.

## Important limitation

The GCN input explicitly contains prompt/response role, absolute position and response position. Its edge weights also average attention over heads and layers. Therefore the result does not yet prove that exact attention topology or weight-endpoint pairing causes the gain. The next confirmatory experiment must compare the same GCN training under:

```text
real graph
endpoint-rewired graph
weight-shuffled graph
self-loop / no-edge control
position features removed
```

All variants must use the same source split, seed, training budget and node-only readers.

## Decision

```text
Keep:     first-order GCN as the current strong graph baseline
Stop:     current order-2 DBGNN / De Bruijn adaptation
Next:     isolate topology, edge-weight and position contributions in GCN
```
