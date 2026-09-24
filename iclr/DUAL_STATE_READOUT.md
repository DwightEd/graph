# 无标签当前／持续状态读出 v1

实现前冻结，2026-09-24。不按本轮自然标签改公式、挑窗口、挑头或翻转方向。
入口 `main.py transport-dual`；复用 observable 完整缓存，CPU 运行。

## 为什么另做这个实验

旧 supervised temporal 已经输入 current、past、future、current-past；没有丢掉当前状态。
分类器用 token 分类损失学习联合权重，不保证首错或瞬时信号单独的排名不退步。
旧无监督 raw_route 和 route_offline_mean 则是两个独立分数，没有联合读出。
本实验检验方向性读出的当前信号与持续信号能否互补，不复活通用异常核。

## 固定算法

完整缓存每层、每个物理头的四角色总响应先相干合并来源块，再取范数 E：

    r[t,l,h] = (E_history + E_self - E_source) / sum_role E

全部响应为零时 r=0、校正后u=0，不用不可观测消息生成瞬时方向风险；同时保存响应尺度供检查。
非零但非常弱的响应仍可能受随机probe噪声影响，本轮没有验证其可靠性。它不是事实支持，也未恢复适用证据。
默认 middle 层为 [L//4, L-L//4)，与上一版逐头实验相同；不根据标签选头。

为了不把一个天然关注局部历史的头直接当异常，参考排除待测来源的全部回答。
按 b(t)=min(floor(log2(t+1)),7) 分组，仅用参考来源同位置组的 r 构建逐头经验分布。
每组内参考来源等权，同一来源的所有 token 共享其权重。
F 是带 ties 中点的加权经验 CDF；u=2F(r)-1 保留向历史偏移的方向。
末组覆盖 t>=127，不使用待测回答总长度，不访问待测未来计算因果状态。
来源不足/参考没有对应位置组直接说明缺少参考，不静默用全局分布替代。
参考可包含错误，F 不是正确分布；默认 cohort 留出是无标签传导式诊断。
也支持来源完全隔离的外部 --reference，用于固定参考的部署式评价。

窗口固定16个token：
- causal: 当前及最多15个过去 token；
- offline: 当前、最多7个过去和8个未来 token；两端按实际观测数平均。
每个回答重新初始化，不能跨答。先对每个头的 signed u 求窗口均值，再取正向部分。

    current[t] = sqrt(mean_lh(max(u[t,l,h],0)^2))
    persistent[t] = sqrt(mean_lh(max(mean_window u[:,l,h],0)^2))
    dual[t] = max(current[t], persistent[t])

current、persistent、dual 三项全部保存并独立评价，两种时间模式并列。
max 使当前数值不被平滑压低，但提高其他 token 的分数仍可使当前排名/固定FPR召回退步。
RMS 是固定的正向尾部汇总，并未学习头间条件依赖；不能称为证明多头协同。
u 各头可异号，时间累积在汇总前进行，区别于对最终标量做平滑。
不同来源的经验分布仍可能不匹配，来源归一化可能移除真实信号，需要实测。

标量对照：raw_route、observable_route 各自保持当前值，计算同窗口均值及 max(当前,均值)。
标量对照不拟合参考；只测双状态组合本身，不能冒称逐头方法的成绩。
既有 raw_attention、entropy、旧 route_offline_mean 原样保留。
旧离线均值的窗口由原缓存决定，新同窗口基线另外命名，避免混淆。

## 科学边界

不训练分类器，不读取监督预测，标签只在所有分数落盘后加载。
不将熵高、FFN负向量、低source或状态持续自动叫作幻觉事实。
FFN已进入 response_total；本轮不额外加入其符号分数、来源事实继承或语义门控。
不把 token 标签边界当作模型发现的事件，不清零错误边界。
本轮补足双状态读出；适用性和语义继承仍未解决。
完整逐头自然成绩只有服务器运行后才可汇报；轻量分数包只支持 --features scalar。

## 评价与输出

保留 token AUROC/AP、首错/起点/延续、回答前后半、逐答和来源平衡指标。
全局固定10%排名预算输出各回答命中和正常token，以及每个真实span的覆盖/延迟。
此预算是离线排名诊断，不是已校准部署阈值；正常片段为连续annotation-negative run，
不是控制长度/位置/词汇后的matched normal，不能据此声称已排除表达形式。
原缓存、标签和旧risk保持不变；只写新的输出目录并自动打轻量包。
所有完整缓存结果保存 per-head route、response_energy、校正u和当前/持续状态，
轻量包排除逐头数组；保存协议、参考来源与分布、逐token预测、指标和曲线。

## 四答真实标量对照（固定规则后运行）

输入 readout_heads_v1_review_light.zip，只读取其中5种无监督基线、token身份和配置，
不读取任何 logistic 预测作为输入。836 token、81标错；标签在全部分数落盘之后读取。

| 方法 | AUROC | AP | 首错AUROC（2个） | top84标错 |
|---|---:|---:|---:|---:|
| raw_route | .755425 | .199692 | .716556 | 20 |
| raw causal mean | .779200 | .191457 | .776821 | 18 |
| raw causal dual | .779151 | .209526 | .753642 | 19 |
| raw offline mean | .784580 | .193649 | .707947 | 12 |
| raw offline dual | .789290 | .217929 | .754305 | 20 |
| observable_route | .753070 | .205930 | .743709 | 21 |
| observable causal mean | .791939 | .198887 | .785430 | 16 |
| observable causal dual | .782111 | .213081 | .776159 | 19 |
| observable offline mean | .797204 | .203436 | .724503 | 14 |
| observable offline dual | .796354 | .224504 | .776159 | 20 |

当前＋持续不是保证所有指标都增益。observable offline dual 的整体AUROC/AP高于当前，
但最高分84个命中20，低于原当前21；7个连续正常run有5个至少报警一次。
三个错误span中，第一个15token span仍全部未进入该预算；另两段分别命中14/53和6/13。
同答12045 AUROC/AP为 .762222/.477166，12219为 .879507/.189314。
相对各自mean，dual的AUROC略降、AP上升，实际是权衡，不能称所有维度获益。
以上只是标量双状态规则；逐头校正版本仅通过合成缓存和科学有效性测试，未取得自然成绩。
这四答此前已用于设计，不能把本轮称为独立确认；未根据这些成绩更改冻结公式。

## 运行

```bash
git pull --ff-only origin main &&
bash experiments/native_support/run_dual.sh
```

默认读取 outputs/native_support_ragtruth4/observable_transport_v1 的完整逐token缓存，
写 outputs/native_support_ragtruth4/dual_heads_v1，并生成同级 _review_light.zip。
不加载大模型，不训练分类器；要求新输出目录，可用 --output 另指定。
有独立完整参考缓存时追加 --reference 路径；参考模型及采集协议须相同、来源不重叠。
轻量包只运行可用的标量对照：

```bash
python -u main.py transport-dual \
  --input outputs/native_support_ragtruth4/readout_heads_v1_review_light.zip \
  --output outputs/native_support_ragtruth4/dual_scalar_v1 --features scalar
```
