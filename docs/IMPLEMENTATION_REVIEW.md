# 实现核验（2026-09-11）

范围：将 graph 的默认流程改为固定候选条件路径算子；保留 factorial 和 retrospective onset 的既有消费者，移至明确命名的基线入口。没有新增检测器训练或模型干预前置要求。

## 已执行的验证

测试环境：Python 3.11.15、PyTorch 2.13.0+cpu、transformers 4.57.1、NumPy 2.4.6。模型依赖通过本地隔离目录提供，未修改共享环境版本。

```powershell
$env:PYTHONPATH='D:\projects\python_projects\.codex_tmp\reanchor_review_20260911_deps'
& D:\projects\python_projects\.audit_envs\llm_state_lab_py311\Scripts\python.exe -m pytest -q --show-capture=no
& D:\projects\python_projects\.audit_envs\llm_state_lab_py311\Scripts\python.exe -m ruff check main.py route_graph control_graph tests
& D:\projects\python_projects\.audit_envs\llm_state_lab_py311\Scripts\python.exe -m ruff format --check main.py route_graph tests/test_path_operator.py tests/test_route_capture.py tests/test_route_data.py tests/test_route_detector.py tests/test_route_evaluation.py
git diff --check
```

结果：当前 33 项测试通过（补充远端完整脚本测试后）；lint、所改新代码格式和 diff 空白检查通过。Git 提示既有 Windows CRLF 转换策略，无 diff 空白错误。

关键覆盖：

- 相同角色／距离分配和原始特征值仍能产生不同候选端点响应；singleton 组不制造残差，角色常数信号的一阶残差为零。
- 两步 source-through-history 路径按真实层顺序读取，并排除 query 自环。
- 本地随机 tiny Llama 的 float32、bfloat16 原生 forward；包含首 token；改变预测目标及后缀不改变该位置的候选和表征。
- prepare → extract → detect → evaluate 完整公开命令，无测试 mock 替代模型。
- 改变原始 labels/quality 不影响准备结果；构图输入拒绝标签字段；参考来源与测试来源隔离；匹配条件不足时失败。
- 所有表征共用 active mask；参考常数坐标不放大微小未见差异；验证参考流和测试流都完整，即使篡改后重新生成了自洽文件 digest。
- 独立标签评价识别 token 0 的 onset/first error，检查字符范围及响应 digest，并给出来源配对 AUROC 差。

另外执行：

```bash
bash -n scripts/run_graph_anomaly.sh
bash -n scripts/run_factorial_baseline.sh
bash -n scripts/run_attention_audit_evaluation.sh
bash -n scripts/run_onset_choice_audit.sh
bash -n scripts/run_route_evaluation.sh
bash scripts/run_factorial_baseline.sh data/pilot_factorial_events.jsonl outputs/path_residual_20260911/factorial_baseline
python main.py prepare --dataset D:/projects/python_projects/RAGTruth/dataset --task QA --generator llama-2-7b-chat --max-sources 32 --output outputs/path_residual_20260911/qa32.jsonl
```

五个 shell 脚本逐一通过语法检查；历史 factorial smoke 产生 8 张图、拟合 6 个来源、评分 2 个来源；真实 RAGTruth 准备得到 32 响应／32 来源，参考 16、测试 16，无标注筛选。

新增远端完整脚本通过本地 Git Bash + 真实 tiny Llama 子进程验证：从任意工作目录启动、带空格的路径、四阶段完成后写 COMPLETE、拒绝覆盖、提取失败后停止评分／评价并保留异常日志。没有在远端 GPU 上实际运行。

## 五轴自审

- 正确性：预测 y_t 只使用 q=prompt_tokens+t-1 的信息；两步量解释为路径残差，不解释成纯 interaction；强零模型的条件为角色／source unit／log-lag／self。
- 可读性：main 只解析、构造、调用和输出；算法集中在 operator，参考集中在 detector；旧基线入口与新路径都有真实消费者。
- 架构：无 registry、factory、训练框架或自编码器；现有来源 bootstrap 指标被复用。
- 边界：本地模型，不下载／执行 remote code；无标签契约、摘要绑定、完成 manifest、完整流和 source-disjoint 检查；输出拒绝覆盖。
- 资源：只将所需连续层段复制到 CPU，及时释放其它 forward 输出；末步只读 query 行，null 通过前缀和作用于信号，避免额外 dense null adjacency。原 forward 仍保留稠密 attention 峰值开销。

## 未验证及适用范围

没有运行真实预训练 Llama 权重上的新方法 RAGTruth 检测结果；tiny Llama 和 factorial smoke 都不是有效性证据。未运行 GPU 性能／长上下文压力测试；配置中的 attention 大小仅为部分内存估计。当前原生提取支持 Llama，原始数据准备支持 QA、Summary；正确答案候选覆盖未知。

强零模型可能使残差失去有效变化；候选 proxy 和 probe/generator 差异可能削弱辨别力；reference 分布可能稀疏或被异常污染。共享尺度和 singleton 保留都不能消除所有精确位置混杂。这些属于必须由自然数据对照检验的方法风险。
