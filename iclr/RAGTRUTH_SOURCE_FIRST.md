# 面向全量 RAGTruth 的来源优先 token 检测

2026-09-25更新：用户已完成全量运行并上传汇总。开发选择的test AUROC为
Data2txt .784388、QA .878533、Summary .759659，具体基线与内部定位分析见
[来源排序与token细化](RAGTRUTH_TOKEN_REFINEMENT.md)。下文四答与“未全量运行”描述保留为初版记录。
后续CPU读出使用 `bash experiments/native_support/run_ragtruth_refine.sh`，不必重跑此采集入口。

## 本轮决策与最好结果

用户当前优先目标是提高 token AUROC，并在全部 RAGTruth 按 QA/Summary/Data2txt 验证。
按输出与评价粒度，给每个 token 一个风险分数就是 token 检测；同片段共享分数也是合法方法，
但不能据此声称能区分片段内部的错误词与正常词。此前将这种方法排除在“token级”之外并不准确。

用户返回的 carrier token v2 四答结果：

| 类型 | 方法 | token AUROC | AP |
|---|---|---:|---:|
| 单元均值广播 | source_local_unit_mean | .914856 | .451496 |
| 单元内部有差异 | unified_local_anchor | .891718 | .415677 |
| 新软约束完整模型 | unified_token | .866748 | .341315 |
| 新软约束去图 | unified_token_no_graph | .881694 | .367112 |
| 新软约束随机图 | unified_token_random_graph | .872962 | .343077 |

完整模型不如去图，随机图也更好；默认全量主线不再运行图正则和逐 token 删边。
原有测量、图和失败结果保留在原入口，作为机制研究与历史对照。
本轮保留最强固定主候选 `source_local_unit_mean`，新增一个简单的路由收缩系列，
并提供明确标记的 train 开发集选参。没有把四答上选出的权重宣称为全量最优。

## 观测与读出

观察者对原回答 teacher forcing；允许与原生成器不同。原 token 不被替换，不生成新回答。
文本单元仍用原来的标点／换行划分，最长64 token；不是金标、语义命题或推理步骤。

对同一个原 token `y_t`，测量有来源／无来源、完整历史／单元内部历史四种 log probability。
无来源世界只删除 prompt 中已识别的来源块 token，保留问题／任务指令；位置随删除重新计算，
所以来源概率差并非已经隔离位置效应的事实因果效应。

\[
g_t=-\left[\log p(y_t\mid E,H_{u,<t})-\log p(y_t\mid H_{u,<t})\right],
\qquad a_u=\frac1{|u|}\sum_{t\in u}g_t.
\]

固定主分数是 `a_u` 广播到各 token。局部重放削弱远历史的影响，单元均值减少 token 抖动；
这些是检测设计依据，不表示低来源作用必然为幻觉。

路由沿用旧 `raw_route` 的物理头消息范数公式：
先算每条消息 `a_(t,j,l,h) * ||W_O(l,h) V_(j,l,h)||`，
逐层计算“历史消息范数减来源消息范数”占总消息范数的比例，再对层平均。
不能用先平均 attention 的结果代替这些消息。新增紧凑实现没有改这个公式。

分别在每个 task、official split 的未标注观测上，做来源等权经验 mid-CDF：
单元来源得到 `A_u`，token 路由得到 `R_t`。同一个 source 的全部回答共享一个总参考权重。
计算一个统一的候选族：

\[
s_t^{(\lambda)}=A_{u(t)}+\lambda\left[R_t-
\frac1{|u(t)|}\sum_{j\in u(t)}R_j\right],
\quad \lambda\in\{0,.1,.25,.5,1\}.
\]

来源确定主要排序，路由只做可控的内部修正。没有图、等权混入更弱的 selected 来源、TV、
熵真假规则或高维异常模型。`lambda=0` 与来源单元均值保持同样的排序；
非零权重允许同单元 token 不同分，但没有保证它能提高混合单元定位。
这里刻意收缩旧读出中强度为1的路由修正，不靠加复杂模块追求 AUROC。

这组分数仍依赖完整回答与未标注 task/split cohort，是 **transductive 离线评分**。
经验秩不是错误概率。数据集之间不拼接尺度后挑最好成绩，主要报告各 task 的 test 指标。

## 全量采集为何不必跑旧链条

直接从官方 `source_info.jsonl`、`response.jsonl` 准备数据，默认：

- QA、Summary、Data2txt 全部任务；官方文件中的全部生成器；train/test 都处理。
- 没有默认4答上限、正负平衡抽样、按幻觉标签挑回答或每 source 只取一答。
- 保存原始 response/source ID、task、generator、split 与完整 token 数量。
- 空 token 回答明确列入排除清单；超模型上下文直接报错，不静默截断。

两个来源世界各做一次分块 prompt prefill，复用 KV 计算完整历史与各局部单元。
只有有来源完整历史分支额外捕获 attention，以每个物理头的投影 Gram 计算消息范数。
缓存只需保存逐 token 似然、路由及熵；没有原生反传、Fisher/Jacobian 或逐边删除。
每个答案的有来源／无来源完成标记分别保存，中断后重用已完成的世界。

默认不保存庞大的逐头中间数组；计算本身始终区分物理头，最后按已验证的旧公式读出。
需要后续头分析时加 `--save-heads`，保存 token×layer×head 的来源／历史／总消息范数和
来源／历史 attention mass。它们是范数之和，不是向量和的范数，也不是事实支持标签。
没有保存全长逐 key DAG，不能从该轻量缓存恢复任意历史边干预。

似然采集保留原 SDPA 局部分支；有来源完整历史使用 eager 以读取原生 attention。
三类小模型 float32 已与独立原生前向／旧路由公式对齐；未声称真实8B bf16跨kernel bit-exact。

官方格式与 task/model/split/source 字段依据：
https://github.com/ParticleMedia/RAGTruth ，论文 https://aclanthology.org/2024.acl-long.585/ 。
全部数据量以用户当前官方文件的实际行数为准，不写死某个版本的行数。

## AUROC 优先的可选选参

`--select-on-train` 明确是 **使用标签选择超参数**，不是严格无监督结果；不训练分类器。
固定无标签基线及所有候选仍完整报告，选择后分数命名 `selected_detector`，不冒充无标签成绩。

1. 每个任务只用官方 train source，按固定 seed=42 的 hash 排序，取20%作为开发来源。
2. 同 source 的全部生成器回答一起进入或离开开发集。检查官方 train/test 来源互斥。
3. 在开发 token 上选 pooled AUROC 最高的读出。候选除路由收缩系列外，保留
   full/pair来源单元均值、raw_route及其离线均值，允许Summary/Data2txt选择不同的有效基线。
   并列时按预先声明的池顺序：零修正、既有基线、递增路由权重，不能根据test改变顺序。
4. 将每个任务选择的同一读出应用到该任务全部 test 回答，不按 test generator 再选。
5. `selection.json` 保存所有候选成绩、source/response ID、权重、标签使用性质。

其余 train 来源不训练分类器；它们参与各自 split 内的无标签尺度。
来源尺度和路由尺度不共享 train/test 的样本，test 的尺度仅使用其未标注观测。
候选包含零修正，所以开发集选参不会被迫使用降低其 AUROC 的路由；test 没有收益保证。
只有 test 的选择后成绩用于验证，train/开发成绩不能当作泛化成绩。
四答已反复参与设计，包含这些来源的官方test也不能宣称全部未触及的全新确认集。

## 一键运行

脚本默认沿用四答缓存里记录的模型和官方数据目录。

```bash
git pull --ff-only origin main
bash experiments/native_support/run_ragtruth_all.sh --select-on-train
```

这条命令完成全量准备、采集、固定评分、train 开发集选参、分任务评价与轻量打包。
若要求严格无标签，去掉 `--select-on-train`。
模型与数据位置不同可设置：

```bash
MODEL=/path/to/Meta-Llama-3.1-8B-Instruct \
RAGTRUTH_DATASET=/path/to/RAGTruth/dataset \
bash experiments/native_support/run_ragtruth_all.sh --select-on-train
```

默认输出 `outputs/native_support_ragtruth_all/source_first_v1`。
中断后原命令续跑。可降低 `--query-chunk-size 8`、`--prefill-chunk-size 128` 控制运行显存，
这些执行分块允许在续跑时改变；数据筛选、模型、dtype、单元长度等协议不能偷偷改变。
没有在本环境启动全量8B GPU实验，也没有全量自然数据 AUROC。

独立阶段：`--stage prepare/capture/score/select/evaluate/pack`。
`score` 冻结所有分数后停止，不读取标签；`select` 只访问允许的 train 开发标注。
已完成后 CPU 重评，无需模型：

```bash
python main.py transport-benchmark \
  --output outputs/native_support_ragtruth_all/source_first_v1 --stage evaluate
```

复用原四答 contrast/carrier 缓存，不做新采集：

```bash
python main.py transport-benchmark \
  --cache-input outputs/native_support_ragtruth4/message_carriers_token_v2 \
  --output outputs/native_support_ragtruth4/source_first_v1 --resume
```

仅有 test 缓存不能执行 train 选参。新全量缓存是轻量格式，不伪装成旧 carrier 干预缓存。

## 结果文件与本轮实测

- `manifest.json`：全部回答 ID、分组、选择数量、排除记录与数据位置。
- `metrics_by_dataset.csv`：按任务和官方 split 的 token AUROC/AP，以及答内／单元内 AUROC。
- `metrics.csv`：进一步按生成器、all-error/onset/first-error/continuation/前后半段细分。
- `evaluation.json`：完整分组结果与混合单元数；无混合单元时内部 AUROC 为 null。
- `summary.json`：固定主候选及可选选参方法的三个任务 test 摘要。
- `datasets.png`：三个任务的 test AUROC/AP 对照图。
- 各回答 `observations.npz`、`scores.npz`、`components.npz`；选参结果另存 `selected_scores.npz`。
- `*_review_light.zip`：顶层汇总、协议与清单。体积不随逐头轨迹膨胀，**不含全部逐 token 数组**；
  原始逐 token 文件保留于输出目录，可按需要另行提供。

本轮复用已上传 **carrier v1** 四答，836 token/81 标错，实际结果：

| 候选 | pooled token AUROC | AP | 答内 AUROC |
|---|---:|---:|---:|
| source_local_unit_mean / lambda=0 | .914856 | .451496 | .957736 |
| lambda=.1 | .915706 | .376313 | .952360 |
| lambda=.25 | .916049 | .401025 | .944071 |
| lambda=.5 | .910604 | .412074 | .932945 |
| lambda=1 | .883395 | .405836 | .897849 |

这个预先声明的系列中，最高观察到 .916049，且保留 token 内部差异。
相对 .914856 仅提高约0.12个百分点，AP和答内排序下降，不能声称定位已改善或统计显著。
没有据这四答把 `.25` 固定为全量最优值；固定主候选不变，选参仅由 train 开发来源完成。
40项针对性测试通过，涵盖三类小模型、旧路由一致性、全部任务/生成器/split、断点重用、
缓存不改写、零权重基线排序不变、train/test source 隔离和改变 test 标签不影响选参。
