# P7 full development after the independent pilot engineering gate

Do not run until the fresh P7 pilot witness returns actual exit0 and passes its
bounded numerical/provenance checks. Preserve both old P5/P6 failed validations.
The scorer/protocol are unchanged from pilot: f6bb3332d3743733d5d8337e6753978c4ff7ded4a2d6476121bfd9f10a54589f
and d54fa4b6c6e91f6868d37e6eb65eb59db5db5840af54b21ee17d7313d2798a5e.

Use the existing research@03909e02 environment (environment-spec identity, not
Git SHA), one idle RTX4090, same model files and FP32 arithmetic. No package
installation, input truncation, changed budget/precision, sample removal or retry
into an existing output. Full original development32/16sources/5170tokens only.

Exact command:

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/graph/scripts/run_local_grounding_reasoned.sh development /share/home/tm902089733300000/a903202310/lys/research/graph/outputs/p7_reasoned_development_20260914_v1
```

Main launcher launch_p7_development.py additionally checks the pilot actual exit,
complete2/2 hashes,8 canaries, fixed scorer/protocol, environment/version/model
metadata, idle GPU and absent output/log/launch record. It starts that exact
command once in a detached Python supervisor. Actual child returncode and UTC
timestamps are recorded in outputs/P7_DEVELOPMENT_LAUNCH_20260914.json;
stdout/stderr in runs/p7_reasoned_development_20260914_v1.log.
Consult this record and live process before resuming; never launch twice.

All32 score files must be complete/hash-frozen before the separate CPU evaluation.
Use next_iteration.development_assess_scoped --primary reasoned_source_risk,
with complete P6/P5/population reference predictions. That evaluator only admits
the original32 IDs and parses their annotation rows; no new128 annotations.
Its full P6 regression, including500-bootstrap objects, matched exactly.

Development is selection evidence, not goal achievement. The next confirmation
roster is outputs/p7_confirmation_roster_20260914_v1, fixed64sources128answers,
input SHA cb16edbc11ad1d5cf637481d038d2ac2bec0f72136df97fc461103b9f06d196e.
It excludesallR04/spentP6 sources and exactsource texts and was built before P7
natural efficacy. It still needs a separate frozen method/confirmation interface;
do not run it with this development command or reuse old validation labels.
