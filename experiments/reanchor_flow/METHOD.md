# Method

For response token `p`, the causal predictor is `q=p-1`. For layer `l`, query head `h`, and
source `s`, define the transported-message capacity

\[
c_{l,h,p,s}=A_{l,h,q,s}\lVert W_{O,l,h}V_{l,g(h),s}\rVert_2,
\qquad
P_{l,h,p}(s)=\frac{c_{l,h,p,s}}{\sum_u c_{l,h,p,u}}.
\]

The audit separates two internal event types.

## Prompt revisit

\[
R_p=\mathbb E_{l,h}\sum_{s< P_0}P_{l,h,p}(s),
\qquad
\Delta R_p=R_p-\operatorname{median}(R_{p-w:p-1}),
\]

where `P_0` is the response start. This asks whether the current prediction renews transport from
any prompt token. It does not require the prompt source to exceed a distance threshold.

## Nonlocal review

For lag `d=q-s`, use the continuous weight

\[
w(d)=\min(d/D,1),
\qquad
N_p=\mathbb E_{l,h}\sum_s P_{l,h,p}(s)w(q-s).
\]

`D` is `distance_scale`. Thus nearby sources receive small weight and increasingly nonlocal
sources receive larger weight; no source is classified by a hard far/near gate. The event signal is

\[
\Delta N_p=N_p-\operatorname{median}(N_{p-w:p-1}).
\]

Prompt revisit and nonlocal review are intentionally distinct: a token can inspect an older
response state without returning to the prompt, or return to a prompt token near the prompt end.

## Route change and future anchor

Route change is the Jensen-Shannon divergence between the current full source distribution and the
mean of the preceding `w` distributions. Future influence is the mean normalized message mass that
later prediction events assign to token `p` within horizon `H`.

Prompt-revisit peaks, nonlocal-review peaks, and future-anchor peaks are selected independently as
positive local maxima above a frozen within-sequence quantile. Each event type is coupled when an
anchor occurs within `0..max_lag` tokens. Circularly shifting the anchor sequence gives a null.

Population evidence is reported in two ways:

1. pooled event coupling minus its pooled circular null;
2. the source-level mean of per-sample coupling lifts, with a source-bootstrap 95% interval and the
   fraction of sources whose lift is positive.

The second quantity decides whether the rhythm is broadly repeated rather than driven by a small
number of long responses with many peaks.

Punctuation never selects peaks; it is only a visual reference. Capture is label-free. Labels are
opened afterward to compare the first hallucinated token with a nearby clean token in the same
response. Future influence at an onset is post-outcome evidence that an error becomes reused; it
cannot by itself explain why the error was generated.

This is an observational mechanism-discovery audit, not yet a causal circuit or detector. Targeted
path interventions are justified only if a coupling lift is positive across independent sources.
