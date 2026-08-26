# GroundedRoute coding contract

Before changing code in this directory, read and follow:

```text
iclr/ENGINEERING_RULES.md
```

The rules are part of the project contract. In particular:

- keep one official implementation on `main`;
- keep modules small and named by research step;
- avoid duplicate versions, defensive hash/schema chains, and oversized result dictionaries;
- preserve exact layer/head/endpoint structure until graph aggregation;
- aggregate neighbour and edge information into `node_embedding`;
- keep downstream detectors node-only;
- keep labels out of representation learning and unsupervised detection;
- provide a direct one-command shell runner without `set -euo pipefail`;
- record hypotheses, commands, commits, results, controls, and stopping decisions.
