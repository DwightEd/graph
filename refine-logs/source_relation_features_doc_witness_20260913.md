# SourceRel-Mini 独立文档执行见证（2026-09-13）

执行者：fresh agent `sourcerel_feature_doc_witness`。按 `docs/SOURCEREL_FEATURE_RUN_20260913.md` 逐字、顺序、各一次执行前三条命令；无安装、环境重建、代码修改、覆盖或重试。环境 spec canonical hash 为 `03909e02`，与现有 ledger 一致。执行前与 sanity 结束后 GPU 均为 0 号卡 1 MiB / 24564 MiB。

| 文档阶段 | 实际 session | PID | exit |
|---|---:|---:|---:|
| 全量 CPU prepare | 66930 | 169275 | 0 |
| 两来源 CPU sanity prepare | 91134 | 169421 | 0 |
| GPU sanity encode | 57496 | 169518 | 0 |

全量 tokenizer 预检：883 来源、7064 query、38541 唯一视图、3954619 tokens；最短44、最长1200 tokens。超4096或其他不可编码视图0，受影响 query pool 0。无截断。全量 output 为 `outputs/source_relation_features_v1_20260913`，prepare manifest SHA256 为 `ac4e920b4d21d5bfcf99e165c9c923a0fedd53f499a6735b8c3f0906fded625a`；settings SHA256 为 `48cc58fa5854ab4d3891f601ca8fd501c97ef5f4357131b833f02509dea1d615`。

Sanity 由预先固定的 source 13599（source_train，8 query、35 candidate）和13610（source_validation，8 query、30 candidate）组成，共81唯一视图、9044 tokens，最短64、最长402 tokens。实际21 forward；编码循环stdout计时1.6575267501 s，summary的 `seconds_excluding_model_load=23.4404536523` 包含结束前完整输入/代码/模型校验，不能当成纯GPU时间。CUDA峰值16256140800 bytes（约15.14 GiB）。没有外部统一wall timer，不虚构prepare或总进程耗时。

Sanity 最终array为81×4096 float32，非有限值0，非零元素331429。实际从bf16因果模型最后一层、最后非pad token提取辅助canonical-view表示，并保存float32；它不是原生成时的隐藏状态，不产生事实真假判断。array SHA256为 `f6b43bb48efce6f240aa33a308a828da31f48e14f806e4b62b28f584b4710f5d`，最终manifest SHA256为 `e6a22891484b9793f1644c490b9ed2335d6132a70e4537de3621ee549d9e936c`。

完整input/view/code/model/array hashes及独立核验记录保存于 `refine-logs/source_relation_features_doc_witness_evidence_20260913.json`。Sanity完成后已通知training witness做独立head sanity；该代理报告session25965 exit0。本见证仍等待root明确确认后才执行文档第四条全量encode命令。当前没有文档与实况偏差，没有自然归属有效性或完整检测器收敛结论。

## 后续全量 encode 已完成：覆盖前述待运行状态

Root明确核实head sanity session25965 exit0后，本fresh witness按原文档第四条命令执行一次：session62985，PID169859，实际exit0。没有重做前三步、安装、代码修改、覆盖或重试。全量38541/38541视图成功，9636实际feature forwards，3954619 tokens，unavailable 0。

编码循环实际stdout计时 474.0549100935 s（约8342.11 tokens/s，81.30 docs/s）；summary计时497.9294861834 s包含结束前完整校验。CUDA峰值16642519040 bytes。最终array shape 38541×4096 float32，非有限值0，非零元素157713201，文件631455872 bytes。

独立后验重新核验全部50个live与snapshot代码文件、16个模型/配置/tokenizer文件、完整人口输入SHA、父source数据的全部manifest artifacts，以及feature的全部prepare/final manifest artifacts均一致。Array SHA256为`6934abb01ff9ab0c4f293bfd044a34bd0b538fdb7a96b2c6a95f65ff7ba750a0`；最终manifest SHA256为`d2405dc13eccc9ca8ef3be857ce39f931860f88416e771bdc6ea60e008a43539`。

完整后验机器证据：`refine-logs/source_relation_features_full_evidence_20260913.json`，SHA256 `81a99a221f937120b9a03c9b00171233e9e0cdd35cc3fbb5cdd7fc821e3a2b72`。189条实际stdout采样记录另存`refine-logs/source_relation_features_full_progress_20260913.json`，明确不是完整stdout转储。完成manifest、exit和array核验后已通知root及training witness可接续full head训练。文档与实际无偏差；本报告仅证明特征执行和完整性，不证明自然约束归属有效或方法收敛。
