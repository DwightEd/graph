# 联合模型阴性结果、读出修正与 RAGTruth 扩展验证

2026-09-22。本轮先固定已失败的假设，再检验一个有限改动；不重新堆叠融合器。

## 四答真实结果

输入已经是 RAGTruth QA/test/llama-2-7b-chat 的真实回答，由 Llama-3.1-8B-Instruct 回放。
836 token、81 错误、3 个段起点、2 个每答首错；四答按标签平衡选择，属于已反复使用的诊断集。

| 方法 | AUROC | AP |
|---|---:|---:|
| 原功能路由 | 0.754983 | 0.199373 |
| 普通均值 | 0.779200 | 0.191355 |
| joint_state | 0.756062 | 0.170401 |
| route_state | 0.755392 | 0.172340 |
| instant_state | 0.747265 | 0.196363 |

联合模型相对普通均值的总体/同答 AUROC 分别下降 0.023138/0.035394。
相对单观测 route_state 仅增加 0.000670，AP 还下降 0.001939。
因此尚无多观测联合建模增益；不能把模型架构完整当作检测有效。

## 能直接定位的问题

从用户提供的四个参考协方差得到 corr(R,A)=0.983640、0.981468、0.990806、0.982766。
说明两个观测高度冗余，但不能据此断言差值/条件残差没有信息。
A/H 只影响原实现的段长度后验，不直接提供新的风险方向；原模型不是三个检测器的风险融合。

原读出对每个可能长度 n 使用 mu_R=(kappa*mu_0+sum_R)/(kappa+n)。
瞬时对照因此是 (R+mu_0)/2；同答排序不变，不同 source 的 mu_0 却会改变跨答排序。
这是先验收缩的建模选择，不是实现算错。它在该瞬时对照中降低了 pooled AUROC，
但不能仅据汇总结果断定联合模型全部退化都来自它。

## 唯一修正：固定分段，分数只平均已观察路由

    S_observed(t) = sum_n q_t(n) mean(R[t-n+1:t+1])
    prior_pull(t) = S_joint_state(t) - S_observed(t)

q_t(n) 直接读取 v4 保存的 run_posterior，不重算分段，不改变 covariance、hazard 或窗口。
这是状态长度后验加权的经验段均值，不再声称是原 NIW 模型的后验均值。
只去掉风险读出中的直接先验收缩；参考数据仍影响 q，未排除全部参考分布影响。
新评分仍是因果路由平均，并非注意力/熵直接融合，也没有恢复语义适用性。
它用于检验和纠正已定位的问题，不宣称创新或 AUROC 必然上升。

不载入模型、不访问原始大缓存：

```bash
python -u main.py support --stage readout --output outputs/native_support_ragtruth4
```

输出 state_readout_v5/w16/report.html、evaluation.json、comparisons.json、tokens.csv、NPZ。
原 v4/v3 全部保留；候选 joint_observed 与原路由、普通均值、原联合模型固定比较。
tokens.csv 增加 prior_pull 和 observed_current_weight；不根据这四答结果调参或挑最佳窗口。

## 扩展到新的真实数据

一次先做 16 个 train source 的无标签参考、32 个 test source 的评价；不启动全量数据。
选样本只看 task、generator、split、source ID 及固定 seed，不看标注是否为正。
按 sha256(seed:source_id) 排序，每 source 取一个固定 ID 回答。
参考、测试、--exclude-output 提供的已检查 source 三者不相交。
这仅认证排除了明确提供的历史样本，不能认证这些 source 从未在项目其他实验出现。

```bash
git pull --ff-only origin main
python -u main.py support --stage validate \
  --dataset /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset \
  --exclude-output outputs/native_support_ragtruth4 \
  --reference-count 16 --limit 32 --query-chunk-size 8 \
  --output outputs/native_support_validation32 --resume
```

默认 task=QA、split=test、generator=llama-2-7b-chat；observer 沿用已有默认模型。
validation_plan.json 在前向前保存 source/回答 ID、排除集合、seed、window 与固定方法。
添加 --prepare-only 可以只准备和核对样本，不加载模型权重；正式运行删去它，保留 --resume。
这会采集新的回答：reference/ 是 train 缓存，test/ 是 test 缓存，已有逐词文件会复用。
采集沿用 teaching 的分块原生前向和现有 NPZ 格式，仍保存完整原始数组；没有承诺新的显存/磁盘压缩。

参考统计在 test 评分前冻结，test source 不参与参考。标注仅用于准备对齐与评分后的评价，
不用于选样、训练分数或挑参数。reference 的标注也不进入参考均值和协方差。
结果：test/state_readout_v5/w16/report.html；根目录 summary.json 给出本轮摘要。
同时报告总体、同答、source 等权、首错/起点/延续、前后半段和恢复位置。

新 cohort 上只作同样本方法比较，不把它的绝对 AUROC 与旧四答直接相减。
若 joint_observed 未稳定超过 route_mean，继续以普通均值作有效基线，不自动宣布联合模型成功。
当前实现没有源级置信区间，32-source 结果仍是扩展诊断，不是最终性能确认。
本地没有用户服务器的模型、数据或 836-token 缓存；本轮没有新的自然 AUROC。
