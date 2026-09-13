# Surface owner v1：完整负结果与方法修订

2026-09-13。完整36回答、6个官方train来源；该清单已用于迭代，不能当未见确认集。结果 `outputs/surface_owner_v1_20260913`，诊断 `outputs/surface_owner_v1_diagnostics_20260913`，运行见证 `refine-logs/surface_owner_v1_doc_witness_20260913.md`。

| 项目 | 实际结果 |
|---|---:|
| B / C 完成回答 | 36 / 36 |
| 目标槽 / 进入C+N门限 | 1619 / 44 |
| 验证候选B/A | 145 |
| 全部门通过B/A | **0** |
| 特征文档 / 实际特征forward | 2377 / 595 |
| 实际reader forward / 返回 | 2199 / 2344 |
| reader缓存命中 / reader_error | 145 / 0 |
| native forward / 标签评价 | **0 / 未执行** |

按首个拒绝原因计数：edited full base124、preservation16、value-only3、selected owner2。重新核验全部门，非互斥失败分别为124、125、145、51；不能把124全部归因于父句含多事实。词面替换含类型/语法不合格者，reader也有明确语义误判。所有候选与拒绝均保留，没有调低门限后报成功。

`refine-logs/surface_reader_failure_review_20260913.md` 独立只读复核未发现实现必修问题：缓存、token、请求与计数可追溯。已观察到明确来源关系被误判或无关字段被判相关，最终门失败关闭，所以没有把错误候选交给native。

为检验有限首token输出是否隐藏格式/质量问题，按五类检查各固定哈希最小两请求，在CPU重放共10请求、30实际forward。见 `outputs/reader_output_cpu_audit_20260913` 及 `refine-logs/reader_cpu_doc_witness_20260913.md`。全部自然输出为允许字母加EOS；允许标签全词表概率和0.998963296–0.999999821；CPU/GPU argmax10/10一致，条件概率最大绝对差0.0458641，因此不是数值精确复现。这个有限诊断排除这些请求的格式/标签质量解释，不建立语义准确率。

据此停止原样增加Qwen有限标签批次。新的结构实现见 `next_iteration/fact_scope.py`、`fact_scope_mask.py`：原文可重叠事实成员、共享变量、三值组合表和坐标级留值遮蔽；它们只在调用方提供的假设下求解。typed Data2txt来源代数作为先行provider，提供可核验的真实单值B/A；通用QA/Summary provider继续保留未解决状态。方法依据和限制见 `refine-logs/fact_scope_graph_revision_20260913.md`。

本批不支持准确自动回看、内部归属辨别、错误路由/聚合失效或连续错误范围。独立工程复核和文档见证不等于科学有效性审计；指定MCP审稿后端未可用，未伪造其通过裁决。
