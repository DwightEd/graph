# Provenance joint state

无自然标签拟合的 token 来源锚定联合 Gaussian 状态模型，显式使用相邻 8 个 token 隐状态与 native 消息边。评分是来源后验均值，不是罕见度。四轮真实实测后，最终 native AUROC **0.563585**，低于 source_unit **0.671476**；真实边增量未确认，保留失败结果。

实现公式：[METHOD_IMPLEMENTATION.md](METHOD_IMPLEMENTATION.md)。逐轮指标、原样例、置信区间及限制：[RESULTS.md](RESULTS.md) / [RESULTS.json](RESULTS.json)。科学代码：`measure.py` → `features.py` → `gaussian.py` → `run.py`；`legacy.py` 提取固定旧对照/来源 anchor，`evaluate.py` 冻结后读官方标签。

## 环境与已有结果复核

从 graph 仓库根目录执行；复用现有 research 环境，不安装依赖或修改模型。

```bash
export PYTHONPATH=.:teaching/state_audit/src
export OMP_NUM_THREADS=4
PROJECT_PYTHON=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
$PROJECT_PYTHON -m pytest experiments/flow_latent/provenance_joint_state/test_science.py -q
$PROJECT_PYTHON -m experiments.flow_latent.provenance_joint_state.audit_numeric --measurement outputs/provenance_joint_state_20261008_measure --runs outputs/provenance_joint_state_20261008_v1 outputs/provenance_joint_state_20261008_v2 outputs/provenance_joint_state_20261008_v3 outputs/provenance_joint_state_20261008_v4_converged --output /tmp/provenance_joint_state_numeric_audit.json
```

原始 v4 token 可视化为 `outputs/provenance_joint_state_20261008_v4_converged/tokens.html`。原缓存/模型/执行代码快照均保留；Git 只包含代码及小型研究报告。

## 重现最后一次优化与评价

新 output 必须不存在，不能覆盖已冻结运行。以下从不可变 v3 继续，复现最终优化设置；rank selection 本身也只用无标签 fit validation。

```bash
$PROJECT_PYTHON -m experiments.flow_latent.provenance_joint_state.run --measurement outputs/provenance_joint_state_20261008_measure --output outputs/provenance_joint_state_reproduce --ranks 2 --seeds 42 43 44 --iterations 200 --encoder anchored --source-gaps outputs/provenance_joint_state_20261008_v1/legacy_source_gaps.npz --view self --head-weight 0.0009765625 --mask-mode sparse --continue-from outputs/provenance_joint_state_20261008_v3
$PROJECT_PYTHON -m experiments.flow_latent.provenance_joint_state.legacy --output outputs/provenance_joint_state_reproduce
$PROJECT_PYTHON -m experiments.flow_latent.provenance_joint_state.evaluate --measurement outputs/provenance_joint_state_20261008_measure --output outputs/provenance_joint_state_reproduce
```

## 完整采集与前三轮命令

仅在需要新数据时运行采集；本次结果的已有缓存不需要重跑。

```bash
$PROJECT_PYTHON -m experiments.flow_latent.provenance_joint_state.measure --inputs outputs/flow_latent_20261007_capture/inputs.json --output outputs/provenance_joint_state_new_measure --batch 8
$PROJECT_PYTHON -m experiments.flow_latent.provenance_joint_state.run --measurement outputs/provenance_joint_state_new_measure --output outputs/provenance_joint_state_new_v1 --ranks 2 4 8 --iterations 40
$PROJECT_PYTHON -m experiments.flow_latent.provenance_joint_state.legacy --output outputs/provenance_joint_state_new_v1 --sources-only
$PROJECT_PYTHON -m experiments.flow_latent.provenance_joint_state.run --measurement outputs/provenance_joint_state_new_measure --output outputs/provenance_joint_state_new_v2 --ranks 8 --iterations 40 --encoder anchored --reuse outputs/provenance_joint_state_new_v1
$PROJECT_PYTHON -m experiments.flow_latent.provenance_joint_state.run --measurement outputs/provenance_joint_state_new_measure --output outputs/provenance_joint_state_new_v3 --ranks 2 4 --iterations 40 --encoder anchored --source-gaps outputs/provenance_joint_state_new_v1/legacy_source_gaps.npz --view self --head-weight 0.0009765625 --mask-mode sparse
```

以上 v1/v2 rank 命令是本轮选出 8 后的历史设置，不代表未来来源必定选 8。`--reuse` 会检查相同观测字节、尺度、权重与来源身份；不满足必须重拟合。`--continue-from` 还检查完整 feature map。自然检测标签仅在 `evaluate` 加载；`--diagnostic` 另外训练监督诊断，不进入 detector。

本数据为历史暴露的 53 来源，不能用重新执行命令称盲确认。物理端点 replay 门、精确词项匹配 null 和新来源确认未完成。新模型未替换已有强检测基线。
