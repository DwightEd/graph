# 本轮审查状态

外部科研评审：REVIEW_UNAVAILABLE。未获得研究评分、创新性认证或实验完整性通过；result-to-claim 暂记 claim_supported=no，pending Codex review。

第一轮独立工程审查要求修正归档字段/候选/跨回答宽度校验、候选上下文缺失语义、对照独立覆盖、随机端点置换、归档原子标记、校准与分数绑定、报警来源bootstrap、无效标注输入、模板配对汇总和严格zip。已修正并复跑受影响实验；等实际报警预算仍明确未验证。

第二轮工程复审记录于 engineering_review_round2.md。全套软件验证：graph 39 passed；reanchor 55 passed；新增代码ruff检查通过；独立16条件GPU smoke通过。11个关键结果字段存在性核对通过，两个结果包的全部文件哈希核对通过，128机制条件及执行代码/案例哈希核对通过。这些检查均不替代独立科研审计。

当前研究决定及证据见 [完整报告](../docs/METHOD_ITERATION_20260912.md)。

第二轮发现按端点数重复同一置换的问题；改为条件稳定seed并完整v3复跑，追加各方法/两方法有效错误集、评价哈希和全部非图kNN报警配对。第三轮 engineering_review_round3.md 给出 APPROVE，仅限工程修复，无剩余工程阻断。科研评审仍不可用，研究结论仍为未获支持。
