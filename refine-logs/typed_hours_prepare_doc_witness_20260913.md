# Typed hours CPU prepare — independent document execution witness (2026-09-13)

Fresh witness executed the documented invocation verbatim exactly once. Exec session **22144 exited 0**. No command edits, parameter substitutions, implementation changes, reruns, installs, environment rebuilds, model loads, or population pause/resume actions were performed.

Documents read before execution: `docs/TYPED_HOURS_PREPARE_RUN_20260913.md`, `.aris/compute/local.md`, the applied `run-experiment` skill and its compute environment contract. The independent core/prepare engineering report `refine-logs/typed_hours_engineering_20260913.md` records the sole Required provenance finding closed (final Critical 0 / Required 0). Existing `research@03909e02` environment was reused.

## Exact invocation and observed completion

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.typed_hours_prepare --inputs ../reanchor/outputs/ragtruth_population_20260912/inputs.jsonl --output outputs/typed_hours_population_prepare_20260913 --observer-model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
```

The target output directory did not exist at the pre-launch check, 2026-09-13T03:43:31.299359+00:00. Settings and execution snapshots appeared before row artifacts; summary and manifest were absent during the 1,514-row observation. The runner reached all 17,790 rows and published `summary.json` and `manifest.json` with `status: complete`. Runner-reported compile time was **370.4267426598817 seconds**; this excludes import/tokenizer startup and is not presented as total process wall time. There were no stderr errors or doc-versus-reality divergences.

Output: `outputs/typed_hours_population_prepare_20260913`.

## Denominators and typed compile counts

The frozen settings roster contains 17,790 distinct response IDs. The binary input line count is also 17,790, and the row artifact names equal that full roster exactly. No row is removed from the denominator.

| Split | QA | Summary | Data2txt | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 5,034 | 4,758 | 5,298 | 15,090 |
| test | 900 | 900 | 900 | 2,700 |
| Total | 5,934 | 5,658 | 6,198 | 17,790 |

| Count | train | test | Total |
| --- | ---: | ---: | ---: |
| Response words | 1,982,278 | 341,257 | 2,323,535 |
| Task outside typed provider | 9,792 | 1,800 | 11,592 |
| Source schedule unresolved | 3,438 | 540 | 3,978 |
| Typed assessment complete | 1,860 | 360 | 2,220 |
| Base units | 17,304 | 3,310 | 20,614 |
| Fact outside relation/scope grammar | 17,295 | 3,309 | 20,604 |
| Typed contrast available | 1 | 1 | 2 |
| Native inputs compiled | 1 | 1 | 2 |
| Typed supported, no error contrast | 8 | 0 | 8 |

There are **370 unique source schedules** and **2 compiled legal contrasts: 1 train, 1 test**. The test entry was only counted and hashed; no official-test response/source/contrast example content or gold was inspected. The full-run aggregate grammar counts above are frozen execution metadata. They are not annotation accuracy or native mechanism evidence. The natural train population supplies one eligible source/contrast under this frozen grammar; this witness performs no selection or downstream native run.

## Exhaustive artifact and provenance verification

Verification finished at 2026-09-13T03:51:25.208096+00:00 using a separate read-only verifier (exec session 33950, exit 0). All **18,163 manifest artifact SHA-256 values matched**: 17,790 rows, 370 sources, 2 contrasts, and summary.json. All **50 original code files and 50 executed-code snapshots** matched their frozen settings hashes. All **4 tokenizer/config files** and the frozen input matched their settings hashes. Settings file hash and settings object digest were checked with their separate documented semantics. The summary's split counters add exactly to its total counters; native/source artifact counts match its records. All **18,215 output files** are covered by the manifest, settings/manifest themselves, or code provenance, with no missing or unexpected files.

| Identity | SHA-256 / digest |
| --- | --- |
| Frozen input | `be388df51e83f60beafbcb4075bde85da11a40f0941279e715f0f334e28d67bb` |
| settings.json file | `45b4af6d3c163730fc6a8f91bc82b7c130aef283f761c411e8650aaef8953b20` |
| Settings object digest | `224973133762ed3cda588f606c53b875ea9b7088b222ab974b88c80fe53a6ade` |
| summary.json file | `cc48deccebc33bd418f65613e2e74dd0d2fcaee5dc6e427751f02d0b08bbff57` |
| manifest.json file | `724e6b7af90c9f69f12b4191592d0eca24c167d5ea54f6eae29094a07e52184a` |

The complete per-artifact hash roster is preserved in `outputs/typed_hours_population_prepare_20260913/manifest.json`; every entry was independently recalculated, not sampled.

Tokenizer/config hashes, independently matched against the existing observer directory:

```json
{
  "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
  "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
  "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
  "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424"
}
```

Code hashes, each independently matched against both the workspace original and its executed snapshot:

```json
{
  "next_iteration/__init__.py": "29042590fca33545f5f22ed087c1380d811b882e346cd22d38c0329be8c1a39e",
  "next_iteration/graph_boundaries.py": "7e483b460287f039d2faeb2c9275a5256a78244352413560588012fb43784fbd",
  "next_iteration/surface_graph.py": "a8e7458839542ad1e06e5f43076061f0700b2c516647645b15ef48b88b02731c",
  "next_iteration/typed_hours.py": "2d83753a3401b11caa5cf3d69364e72152e2e514cc199c60109bdb2b0c435ca2",
  "next_iteration/typed_hours_prepare.py": "3460b4b90bd7e18f36a970015be09ecce839c99bdd1f9a7d08d85621ac4a7ff0",
  "route_graph/__init__.py": "1437ed67706d60863a2fc8a1ade29775397b171b084b22b385bc70da6804bc02",
  "route_graph/adoption.py": "80b30b25ccc77142a8961d7fb5b9e1e24b3b3809fccd609d7a561966c9c741ee",
  "route_graph/alarms.py": "624dcfa185e37aa2916e10f56241c3dfb29db06b267baea53101f917c9982d61",
  "route_graph/archive.py": "5d23a35601978d5dc9ba583415958d92a5209a473cff48d46fd7a51bcfd2f4a4",
  "route_graph/archive_evaluation.py": "a22df52e77921ec3546e5387d9817632408c6fc4107074597a9df65e4ed7513c",
  "route_graph/atomic_anchor.py": "c4cc98ed6d8174061b0e6a747bec6db9afa4b42b453889e856796960aa0ded53",
  "route_graph/audit_alignment.py": "08d83b490d23388f4d513e75fe80d55f0775ddf3c44c596f8397600134524131",
  "route_graph/audit_artifacts.py": "2bbe08d6258f5e33a90671c7c5e5f735af19722e73690ad22f5899844378b43c",
  "route_graph/audit_certificates.py": "eb77ed76b0d9404a4da6b22ff9a3ef184aa9e11a46964662f57a377603685739",
  "route_graph/audit_interactions.py": "fe26c61c7afd6048b70789eea60254a665b79c566d9eb957f6e472d0ba844cd4",
  "route_graph/audit_output.py": "88e5f8b575e477bae2a604064e0da1a510f990504bc3f367eea11c84604e80fa",
  "route_graph/audit_phase_native.py": "a4f2f79a05f01650e00d5d1f237c1d3a974dc60139b99aa811f0e6d47b7726be",
  "route_graph/audit_pools.py": "c32648acdcc2f4e3efeb969ddfc3fbe89624dc726a58b262c9f3ba59cfc1d83a",
  "route_graph/audit_prepare.py": "e8f5abaa6d6023e57b4a7c82b31fefbec58021fcbefa762eaccba30db0f3b335",
  "route_graph/audit_prompts.py": "5363051fab3dd2686f66e0bcbaca79825eb415c77c61f3e3e3278f03e244f65f",
  "route_graph/audit_protocol.py": "209ac48b66f8955cfe9962688f9c684cbb42df4e68fcaca8b5691fc8538025e4",
  "route_graph/audit_runner.py": "159a8e494c6b5958796117f372d4929815fc35f1e5b040c8c83dd4d809e005f7",
  "route_graph/audit_semantics.py": "02157f94b8c19f414bdd3589a48190d554018b2eaba1af61d2a09b2a3c87991e",
  "route_graph/capture.py": "df723bc27684a68bf70a8b9386acb6f7cabbb75b40a73519f4e96f68186fd115",
  "route_graph/causal_contrast.py": "33fef4ff694673174427c8700b410779805b9f462723664e9b1ad7ec00de38f9",
  "route_graph/causal_groups.py": "9e9ec3ded2475a7a89740e06139ba9d007dbb4ab128605244d4e9f85f87d499c",
  "route_graph/cloze_anchor.py": "909fd922eb6e5f3653f46b8accb0d669b53541a5b4311696d1c6fc478a0f3b96",
  "route_graph/cloze_prompts.py": "43f275b81df5b44a95747fac5d0def1f0e60a17c5fe2552b54debcc59ae41404",
  "route_graph/context.py": "19f78da58243828d196c4b14cb0eacb2b51d485fafc2e4920b7db3024c639b6d",
  "route_graph/data.py": "a1f12fbebe7ecce43ad2375d4c738e193246eeab564330ccb1a4f23fb4c2de62",
  "route_graph/detector.py": "6fd36d1b5ee62a22497abb8ebc81784499d57c5ca82637f219ddebe53ddb68e0",
  "route_graph/evaluation.py": "ccfe35eef48de4aa40b5455962d160bd0be2dc6fa321324576533fce93272bee",
  "route_graph/event_matcher.py": "d332916aa2b3218ed8a33c406adcd0aeb6a36721e81104d3fc2943ec30b98ea2",
  "route_graph/evidence_anchor.py": "a7e71c5101054ad41bff3e8b647a8bb09cfdbe7fc963c7c76474a3bc3fe99b42",
  "route_graph/frozen_reader.py": "72897d0f290594d4143027d3dabb6c18b239a458ee2506ee5e33eddcaf0601de",
  "route_graph/json_framing.py": "d386597e4332d1edef90a157258a269e91d423c6a6530d029199ae25d7ccaf87",
  "route_graph/metrics.py": "e0d9e792bdfb58a4bda0c398100715f88cef5345203f67a1ded5e6af05f72f26",
  "route_graph/native_audit.py": "5873dff6b32c0dc4b27333559586fcb87128fa0bc55cc291933e36e506470068",
  "route_graph/operator.py": "769c13a79b4c4ad7563b9da93de0f4ca22ed503b639803c4fd20ff216cc1e16b",
  "route_graph/ownership.py": "c6c3ad964b8b9687c768c796fe1fa9c1c652e17024095c84f634e36b6bf84661",
  "route_graph/population_mechanism.py": "eff6e89c9dbf328361f80c7a46bd2f537e46c22c7fd675cf4bd5af2e054d8812",
  "route_graph/relation_readout.py": "d544af3287818adf87d02eb2ce9dcdf9ade003877c70999c388de2a8fc1dc877",
  "route_graph/sample_graph.py": "ba7408a1b9b51484997f0fc7324eb3fdb62a004368e2c74634976739b94aee0d",
  "route_graph/soft_graph_energy.py": "940758fcbd4d774b5737851b502ce9567640f557bef0761277fe230c7d4bf433",
  "route_graph/soft_graph_phases.py": "3b43d51c2a9b1871e9055084fa42da2c859008ec0a4e8a4216dcdacfc27fe426",
  "route_graph/soft_graph_runner.py": "ca9706343cab2de588864b1d6d5a8f7fab498b263c5a19262960f0faf6330829",
  "route_graph/soft_graph_structure.py": "e7757773a33b501ec12c5c626d5a04190d650cf52d51915a19c5bb4837fc487a",
  "route_graph/source_event_graph.py": "069e1ba4c1cad1544372ff151ca1981eb3317940fd44b806cc863046d3bdf89a",
  "route_graph/span_feature_capture.py": "51c6ceaeeea6345af397bc807036a9a76fb74b8321dc4535febdbb1fb318cb43",
  "route_graph/target_dependence.py": "50e2360cad5015d8b3e632720d230b15ef1a54f48969b5f2bfb8ac86f2306439"
}
```

Settings/summary/manifest consistently report **model forwards 0, native forwards 0, reader calls 0, labels used/read false**. The executed prepare path loads `AutoTokenizer` locally and offline, performs typed source/response compilation, and never calls a model loader/forward or reader. No GPU work was launched by this witness. This is a CPU documentation/execution and provenance witness, not a new GPU kernel or clean-environment installation witness.

## Original population continuity

Only `progress.json` and the original process state were observed. No signal, lock mutation, pause, restart, or scheduler operation was issued.

| Observation UTC | PID | Completed / 17,790 | Failed | State |
| --- | ---: | ---: | ---: | --- |
| 2026-09-13T03:43:31.299359+00:00, before prepare | 162289 | 16,757 | 0 | running; process R |
| 2026-09-13T03:44:14.793939+00:00 | 162289 | 16,779 | 0 | running |
| 2026-09-13T03:46:35.894114+00:00 | 162289 | 16,853 | 0 | running; process present |
| 2026-09-13T03:50:07.561385+00:00 | 162289 | 16,956 | 0 | running; process present |
| 2026-09-13T03:51:25.208096+00:00, after all prepare hash checks | 162289 | 16,996 | 0 | running; process R |

The original population therefore advanced **16,757 → 16,996 (+239)** during this witness, with unchanged PID 162289 and failed 0. Its final observed remaining count was 794. These are actual progress observations, not an inference from process existence. Population artifact contents were not inspected.

Result: the documented one-shot CPU preparation passes this execution witness. The existing natural roster and all unsupported/unresolved strata remain counted. The two legal typed contrasts do not establish RAGTruth human-label accuracy, semantic generalization, or native causal effect.
