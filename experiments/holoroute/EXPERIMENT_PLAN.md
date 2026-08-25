# Experiment plan

## Primary comparison

```text
flat1024 -> event-only -> +depth -> +query -> +relay -> +holonomy
```

Every comparison uses the same source split, complete-layer masking semantics,
position conditioning and label-posthoc evaluation.

## Required evidence

1. Full HoloRoute improves held-out reconstruction over `flat1024`.
2. Full HoloRoute improves AUPRC over `flat1024` with paired source bootstrap.
3. Depth and query modules each provide an increment over the preceding model.
4. Real relay paths outperform a matched middle-token rewire.
5. Holonomy provides an increment beyond depth and relay prediction.
6. Score-position correlation stays far below the former cumulative rupture
   baseline and the detector beats absolute and relative position.

## Stop rules

- Full ~= flat1024: remove the graph contribution claim.
- No depth increment: remove the multilayer transport claim.
- No query increment: replace the query mixer with local encoding.
- No relay/rewire increment: remove the causal-path branch.
- No holonomy increment: keep the graph completion model and remove curvature
  language.
