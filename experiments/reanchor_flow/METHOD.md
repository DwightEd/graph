# Method

For response token `p`, the causal predictor is `q=p-1`. For layer `l`, query head `h`, and
source `s`, the transported-message capacity is

\[
c_{l,h,p,s}=A_{l,h,q,s}\lVert W_{O,l,h}V_{l,g(h),s}\rVert_2.
\]

After normalizing `c` over visible sources, the audit computes four token trajectories:

1. **Route change**: Jensen-Shannon divergence from the mean source distribution of the preceding
   `w` response events.
2. **Far-prompt revisit**: mass assigned to prompt sources at least `far_lag` positions away;
   `revisit_delta` subtracts the preceding-window median.
3. **Prompt breadth**: entropy of the prompt-conditional source distribution. High breadth means
   broad review; low breadth means selective re-anchoring.
4. **Future influence**: mean normalized message mass sent by response token `p` to the next
   `H` prediction events.

Revisit and anchor peaks are selected independently as positive local maxima above a frozen
within-sequence quantile. A revisit is coupled when an anchor peak occurs within `0..max_lag`
tokens. Circularly shifting the anchor sequence gives the null coupling rate.

Punctuation boundaries never select peaks; they are drawn only as references. Capture is
label-free. Evaluation matches each hallucination onset to a nearby clean token in the same
response and reports source-bootstrap confidence intervals for changes in revisit, route change,
future influence, and prompt breadth.

This is an observational mechanism-discovery audit. It does not yet claim a causal circuit or a
hallucination detector. Path interventions are justified only after a stable revisit-anchor rhythm
is visible and stronger than the circular-shift null.
