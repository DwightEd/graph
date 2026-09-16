"""仅在冻结检测结果后读取金标；本文件不被训练或评分模块导入。"""

from pathlib import Path


def evaluate_saved_predictions(prediction_dir: Path, annotations: Path) -> dict:
    """报告token/首错/延续/片段重叠和边界误差，不能把离线定位叫提前预警。"""
    raise NotImplementedError


def compare_saved_ablation_runs(run_dirs: dict[str, Path], annotations: Path) -> dict:
    """在相同覆盖范围配对比较，按source重采样，不用test选头或调阈值。"""
    raise NotImplementedError
