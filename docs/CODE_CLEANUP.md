# Code retirement, 2026-09-30

The current user requested removal of unused implementations. The dependency closure retains native adapters, full-answer token measurement/readout, automatic-source diagnostics, GSM/history checks, fixed/JS/MMD comparisons, and their actual data/evaluation utilities.

575 tracked files were removed: 450 production Python files, 28 retired shell scripts, and 97 obsolete test files. Mixed test modules retain independently useful native/scoring/alignment checks. The failed token-backtrace reconstruction code was removed from the surviving readout and benchmark modules. Model loading and cache IO no longer import supervised training entrypoints.

[The deletion manifest](CODE_CLEANUP_20260930.json) records every removed file and original SHA256. The remote pre-refactor tree is commit `193d1fdb22b9afd1da227cb22e3adbd3885629ef`; deleted implementations can be read with `git show 193d1fdb:path/to/file`. Older serialized learned models may require that historical checkout; the retained fixed baseline reproduces its scores directly from scalar packs without loading the retired models.

Raw outputs, results, runs, datasets, archives, ignored notebook copies and pre-existing untracked work remain in place. In particular, `experiments/routing_likelihood` was not modified, and its existing utility imports were checked. Pre-existing user deletion of an archive is excluded from this commit.

The historical fixed baseline was reconstructed exactly for all three tasks and all fit/development/test partitions, including all 2700 test answers / 424408 test tokens. This verifies refactor compatibility, not a new method improvement. New-method readiness and the mathematical contract are in [ARCHITECTURE.md](ARCHITECTURE.md).
