"""流程占位：先完成设计和模块，再接入命令行；现在没有可运行实验。"""

from pathlib import Path


def prepare_graphs(cache_root: Path, output: Path) -> None:
    """计划：读取无标签观测 → 原生图 → 片段视图 → 逐样本保存。"""
    raise NotImplementedError


def fit_detector(graph_dir: Path, output: Path) -> None:
    """计划：来源拆分 → 配对对照 → 学习评分器 → 无标签校准 → 冻结。"""
    raise NotImplementedError


def score_answers(graph_dir: Path, model_dir: Path, output: Path) -> None:
    """计划：完整回答离线评分 → 联合边界 → 每个原token及片段分数。"""
    raise NotImplementedError
