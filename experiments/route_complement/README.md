# 完整逐头路由补充实验

已完成48答/6463token。主候选减少部分误报但降低AUROC，不替换原固定无监督来源＋路由。见[实际结果及方法设计](RESULTS_20260928.md)。全部标签只进入冻结后的评价与事后机制诊断，没有真假分类器或标签选头。

从仓库根目录运行（使用现有research环境）：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.route_complement.run --output outputs/route_complement_new
```

需要现有 message_operator_20260928_v2、message_js、source_first、natural traces、transport_topology_cases 缓存及原Llama3.1-8B模型；不是脱离缓存的通用数据集入口。默认48答名单来自operator manifest。约363秒GPU采集、16.06GiB峰值、新增约17GB。不要为查看结果重新采集：已有 outputs/route_complement_20260928/CASE_BROWSER.html 与 diagnostics/HEAD_BROWSER.html；后者41MB内嵌无损数据，现代浏览器需支持DecompressionStream。

入口跳过存在完成标记的阶段，不覆盖原证据。中途不完整capture目录不自动续跑，保留现场后使用新的output路径；已完成输出用于查看，不用于修改参数后隐式重算。diagnose读取局部标签且只供发现先验，不向score反馈选头。

- capture/messages：原生GQA V、完整各头合成向量与Gram；保留全部query/key/head身份。
- score：原norm路由、新net路由、逐头JS及固定来源融合；每任务4fit/4dev无标签尺度。
- diagnose/head_null/head_profiles：全部1024头的局部差异、条件身份随机化、独立无标签fit功能描述。
- verify/test_messages：精确向量重建、范数界、身份/范围及独立数值检查。
- browser：物理头选择与逐token对照，完整源地址数据保存，不以展示top8替代原始数据。

主评分net_route_offline_mean没有保留所有头作为最终分类输入，这是本轮明确测试并失败的标量替换假设；完整多头资料均保留用于下一轮条件联合方法，后者尚未实现。
