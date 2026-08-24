# Experiment Plan

## Goal

Decide whether the proposed dual-axis attention event graph contains enough
position-independent structure to justify the HoloRoute neural architecture.

## Label-free structure gates

### G1: depth continuation

`depth_transport_gain > 0` on held-out source groups. Failure means that explicit
same-pair depth edges are not justified.

### G2: causal relay

`relay_transport_gain > 0`. Failure means that predecessor paths do not predict
a successor event beyond a typed layer mean.

### G3: query coalition

`query_set_gain > 0`. Failure means that a set/hyperedge mixer is not justified;
a local event encoder is sufficient.

### G4: exact middle-token path

`relay_rewire_gain > 0` under layer/role/lag/observation-matched predecessor
rotation. Failure means that exact causal-path identity is not yet established.

### G5: causal-diamond coverage

At least 5% of scored tokens must contain one valid diamond before holonomy is
interpreted. Low coverage is an identifiability failure, not a negative
hallucination result.

## Label-posthoc mechanism hypotheses

- H1: depth and relay errors are higher on hallucinated tokens after nuisance
  residualization.
- H2: relay path dispersion and depth-relay disagreement are higher, indicating
  inconsistent structural explanations.
- H3: diamond holonomy is higher when coverage is sufficient.
- H4: the joint residual score exceeds absolute and relative position baselines.

Both same-token white-box detection and `state[t] -> label[t+1]` early-detection
alignment are reported.

## Mandatory confound checks

1. Spearman correlation with absolute and relative position.
2. Same-response matching by relative position and event count.
3. Cluster bootstrap by canonical `source_id`.
4. Per-task reporting for QA, Summary, and Data2txt.
5. At least five source-split seeds before a mechanism is promoted.
6. The score artifact must store and revalidate response token IDs and the test
   manifest hash.

## Stop rules

- If the joint score does not exceed the position baseline on two tasks, stop
  the attention-only holonomy direction.
- If relay and rewire gates fail, do not build a De Bruijn path network.
- If query-set gain fails, do not add a hyperedge/set mixer.
- If diamond coverage is low or holonomy has no residualized signal, remove the
  holonomy claim and retain only transport reconstruction.
- If attention-only structure remains weak, collect Q/K/V, residual, and MLP
  outputs rather than inventing more attention statistics.

## Next model authorized by a successful audit

A successful audit motivates a trainable event graph with:

- a head-set event encoder;
- relation-specific depth and relay transports;
- causal-path set attention;
- query-set mixing;
- whole-event masking and causal-diamond objectives;
- a position-conditioned one-class density model.

The audit itself is not the final detector.
