# 已退役：来源错接自监督训练

当前不再运行原始/错接配对分类器。它没有先证明与自然幻觉的适用关系失配一致。
窗口枚举、词项平均证据和该代理损失不能作为已发现机制。

新任务：用真实标签做错误/正常的配对机制比较。

```bash
python -m experiments.unsupervised_token_graph.span_audit --help
```

设计：`iclr/MECHANISM_FIRST.md`；实现：`../span_audit/`。
两个旧run入口只给迁移说明并退出，不会悄悄把无标签命令改成有标签训练。
历史data/metadata/evaluation工具保留；模型、对比训练、片段搜索/解码等退出当前目录。
旧代码可从提交 `8e0fcef778d3335f8836c87d1ba7a8be9176612a` 查看。
不删除任何历史缓存、模型和结果，不要求重新前向。
