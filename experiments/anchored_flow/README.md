# 来源约束、信息路由与消息采纳的基线修正

本模块把之前有效的固定无监督来源/路由方法作为主干。来源条件分数提供风险观测；逐token路由提供位置差异；原生消息采纳结构控制相邻位置的局部修正是否应共享。后者不再独立定义真假，也不把整段重复都扩成错误。

## 方法和理论边界

旧基线 b_i=.75 R_pair(S_unit,i)+.25 R_route(R_window,i)。S来自同一原回答有/无来源的logp差，local/full两种历史范围平均；R是原物理头的历史/来源投影消息范数对比。固定秩参考与官方案例旧阈值均复用，绝不把小样本重新校准版本冒充原.890665方法。

新局部观测 v_i=.75 R_pair(S_token,i)+.25 R_route(R_token,i)，使用同一尺度。u_i=v_i-b_i 是对单元广播和16token窗口的局部创新量，正负皆可。沿用旧观测的局限：删来源也改变长度/位置，分数不是事实概率；题目缺失后生成变难并不自动等于遵守全部约束。

对每物理头h、query i、绝对key j，保留 A_hij、g_hij=d(z_y-z_c)/d gate_hij。g包含当前query原生下游，过去KV固定；c是各query自动最强非实际词，未必语义上对应同一选择。定义 E_hi=sum_j|g_hij|，p_hi(j)=A_hij/sum_j A_hij，q_hi(j,+/-)=max(+/-g_hij,0)/E_hi，pi_hi=E_hi/sum_h E_hi。

    w_i = sum_h sqrt(pi_hi*pi_h,i-1)
          * BC(p_hi,p_h,i-1) * BC(q_hi,q_h,i-1)

BC为Bhattacharyya重合系数，不取top头或低秩投影，符号、绝对key、1024头身份全保留在计算中。0<=w<=1由Cauchy–Schwarz直接得出；零作用能量的边为0。它是读取/作用模式兼容度，不是两个token共享真假的证明，也不是已验证跨时间因果路径。完整头级边贡献另存，完整原始key缓存保留。最终w仍是为了优化使用的标量边，不能宣称从未归约。

    r* = argmin_(|r_i|<=.1) .5 sum_i (r_i-u_i)^2
                            + .05 sum_i w_i |r_i-r_(i-1)|
    s_i = b_i+r*_i

这是一套凸的观测校正模型。可等价写成截断高斯观测项和加权Laplace差分先验的MAP，但没有估计真实幻觉后验。严格凸保证唯一解；对偶间隙检验数值求解。u=0时精确退回b；|s-b|<=.1；边为0时两侧互不正则化。有限幅度仅限制改变范围，不能保证AUROC/误报不退化。TV约束修正而非原分数；无需风险起点，可升可降，也允许一两个token的实体变化。是否真的保住事实边界必须靠评价。

## 数据与对照

8个官方旧错例、两组自然配对4答、GSM同题配对6答。自然输入为完整回答，仅31个局部token有已核对真值，其余unknown。GSM全回答计算，按原步骤平均评价首错，后续unknown；不是每个token都有错误标签。

另外8fit+8dev不同GSM问题只建立无标签秩参考和base步骤95分位，原配对问题全部排除。GSM baseline是本轮移植的全32层来源/路由公式，不能称旧单层attention的同一分数。原题是无chat原始prompt；自动保留special/BOS、去掉非special题干形成旧来源对照，不加实体注释。

固定6候选：base、token_observation、clipped、anchored_flow(主)、uniform_tv、shuffled_tv。后两者分别统一边、每query层内物理头打乱；相同幅度/TV系数。所有候选沿用base阈值，因此没有阈值重选掩盖召回损失，也不保证相同实际FPR。必须同时看排名、gained/lost TP、added/removed FP和GSM首错。所有样本已暴露，设计受先前结果启发；评分/尺度/参数不读自然标签，不能宣称盲测或新全测试。

## 运行

在graph根目录，使用已有research Python；无需安装。prepare/capture补观测；edges复用完整消息缓存；score先冻结，evaluate才访问标签。capture真实GPU，edges/score/evaluate CPU。原代码默认和结果不改，输出必须新目录。

```bash
export PYTHONPATH=.:teaching/state_audit/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m experiments.anchored_flow.prepare --output outputs/anchored_flow_new
python -m experiments.anchored_flow.capture --output outputs/anchored_flow_new
python -m experiments.anchored_flow.edges --output outputs/anchored_flow_new
python -m experiments.anchored_flow.score --output outputs/anchored_flow_new
python -m experiments.anchored_flow.evaluate --output outputs/anchored_flow_new
python -m pytest experiments/anchored_flow/test_model.py -q
```

outputs含manifest、原观测、likelihoods、完整逐层头路由量、头级消息重合贡献、六组scores、units.csv、errors.csv、changes.csv。具体成绩以本轮RESULTS.md为准。

## 最终第二轮与复现全部流程

首轮直接token来源创新失败。最终第二轮仅用

    u_i = .25 [R_route(raw_route_i) − R_route(window_route_i)]

替换原u；原来源锚点完全保留。数学求解与边定义不变。`revise.py`保存首轮对照；`controls.py`追加同边权分布的头打乱控制，保持主分数逐元素不变。具体结果和不可达性证明见[RESULTS.md](RESULTS.md)。当前完整检测目标未通过，未替换正式默认方法。

一键复现本轮小样本全流程（约26答新8B前向与完整缓存CPU读出，需现有原始缓存；新前缀）：

```bash
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.anchored_flow.run --output-prefix outputs/anchored_flow_new
```

已完成结果不必重新采集。直接打开`outputs/anchored_flow_20260929_control/TOKEN_AUDIT.html`查看原分数与新分数的逐位置对照。错误步骤中的每个词没有被标成错误，显示的是步骤级标签。
