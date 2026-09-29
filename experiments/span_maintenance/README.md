# 逐 token 的片段维持状态

识别“读取结构持续”这一生成阶段线索，再检验它能否改善检测。不是把稳定当正确或错误，也不是用 gold span 生成状态。本轮复用18答4292 token的32层×32头缓存；代码和两轮实验完成，检测候选未胜旧基线，见[结果](RESULTS.md)。

## 状态如何递推

每头保留 prompt 读取原型、历史绝对地址原型、历史相对距离原型与有效年龄。对历史分支归一化后计算两种 Bhattacharyya 重合：固定位置锚定与随生成平移的续写。prompt 刷新是当前读取减去此前原型的正增量质量，不等于新的语义信息。

    c[h,t] = max(BC(history_absolute, previous_absolute),
                 BC(history_relative, previous_relative)) * (1 - prompt_refresh)
    N[h,t] = 1 + c[h,t] * N[h,t-1]
    prototype[h,t] = (current + c[h,t]*N[h,t-1]*prototype[h,t-1]) / N[h,t]
    B[h,t,j] = product(c[h,u], u=j+1,...,t), B[h,t,t]=1

无历史读取时c为0。c是连续性系数，不是校准概率；N是有效记忆量，不是已发现的语义片段长度。每头的状态先完整保存，最后才汇报分位数/均值。没有将1024头先合成一个attention分布。

成员内读取为sum_j A[h,t,prompt+j]B[h,t,j]。保持log2距离桶、当前词token-ID复现状态和各桶质量，计算端点交换的精确期望。实际减期望是超额读取；纯lag1头必为0。这个控制去除部分近邻偏好，不能排除语法、位置等全部混杂。

索引t预测回答词y_t；历史key prompt+j表示已输入的y_j，只允许j<t。状态不使用未来；软件和真实前缀检查已覆盖此契约。评分复用旧来源观测/参考分布，整个系统不冒称已部署在线生成器。

## 标点的作用

标点是有效但不完美的边界提示。boundary.py仅看已生成前缀的句末/分号/冒号/换行，数字小数点和缩写可能造成错误提示；没有用GSM金标步骤作为状态边界。

- punctuation_causal/offline：纯标点分块的因果/完整块均值。
- hard：内部状态在标点处强制c=0。
- soft：标点处用c²，其他位置不变。读取完全稳定可以跨标点；标点与内部变化一致则加速忘记。这是预定启发式，并非后验推导。

每个版本独立更新原型，不把旧原型上的系数后处理当作新状态。

## 平均与各 token 作用

普通均值对每个输入分数的导数均为1/n，只包含等权影响假设，并未测量某个实体对生成的作用。本实现明确输出两种不同的统计权重：

    state_weights[t,j] = mean_h B[h,t,j] / N[h,t]
    dependency_weights[t,j<t] ∝ mean_h A[h,t,prompt+j] B[h,t,j]
    dependency_weights[t,t] ∝ 1

前者估计持续状态下的共同分数，后者按实际读取偏向历史载体。均不等于已经验证的因果影响，后者当前自权重较大、平滑较少。weights.npz保存完整权重，可逐token审计，不能把这些权重广播成真假标签。

旧base独立保留。raw/causal16/offline16/state_route/dependency_route/head_shuffle均使用逐token来源，仅改变路由；state_both及第二版*_both同时处理来源与路由。所有候选使用旧秩参考、原阈值，未重新调阈值，亦不保证同实际FPR。首轮主state_both、第二轮主soft_both均在各自标签评价前固定，没有自动选优。

正负消息作用分别保存；目标是每位置原词对自动候选的局部margin，过去KV固定，不是根来源或跨离散生成的完整因果传播。payload.py额外读取12个RAG回答的完整逐头128维value合成向量与4096维最终hidden，衡量相邻相对变化；value在W_O前，不能称完整残差消息。它不参与评分。

## 运行与产物

在graph目录使用现有research环境，CPU，无新LLM前向，不安装环境。依赖已记录的message_js、constraint_uptake_dense、anchored_flow_v1、route_complement和message_operator缓存。

```bash
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.span_maintenance.pipeline --output-prefix outputs/span_maintenance_new
```

新前缀避免覆盖。约9分钟，主要时间为全头缓存读取/状态递推；视存储速度变化。单独评价已有结果：

```bash
export PYTHONPATH=.:teaching/state_audit/src
python -m experiments.span_maintenance.evaluate --output outputs/span_maintenance_20260929_v2
python -m experiments.span_maintenance.audit --output outputs/span_maintenance_20260929_v2
python -m pytest experiments/span_maintenance/test_state.py -q
```

- state.py：归一化、两种读取记忆、状态递推、距离控制。
- run.py / boundary.py：真实缓存测量与冻结评分。
- evaluate.py：冻结后读取标签、逐位置错误、AUROC另一秩公式复算。
- audit.py / payload.py：真实前缀、平均污染、标点代理边界、向量变化。
- outputs/.../TOKEN_AUDIT.html：全部回答逐位置审计；GSM按步显示且首错之后unknown。
- outputs/.../STATE_TRACES.svg：读取、刷新与符号模式轨迹。
- states.npz / boundary_states.npz：全部头；weights.npz / boundary_weights.npz：实际汇聚权重。

全部为已暴露开发回归，不声称独立泛化。原数据/原分数/默认入口不变。
