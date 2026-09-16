"""从现有 RAGTruth 文件补齐身份；不使用回答内容、幻觉标签或标注边界。"""

import json
from pathlib import Path

from ..evaluation_data import read_sources


IDENTITY_KEYS = ("source_id", "task", "generator", "split")


def needs_metadata(identity):
    """只在缓存或已有索引缺少分组身份时读取原数据集。"""
    return any(identity.get(key) in (None, "", "unknown") for key in IDENTITY_KEYS)


def find_response_file(cache_root, dataset=None):
    """查原数据目录及其 dataset 子目录；不搜索或改写整个磁盘。"""
    if dataset is not None:
        location = Path(dataset).expanduser()
        if location.is_file():
            return location.resolve()
        folders = (location, location / "dataset")
    else:
        location = Path(cache_root).expanduser().resolve()
        if location.is_file():
            location = location.parent
        folders = []
        for parent in (location, *location.parents):
            folders.extend((parent, parent / "dataset"))

    for folder in folders:
        path = folder / "response.jsonl"
        if path.is_file():
            return path.resolve()

    if dataset is not None:
        raise FileNotFoundError(f"{dataset}: 没有找到原始 response.jsonl")
    return None


def metadata_fields(row, source_tasks):
    """白名单提取；不索引 response、labels、offsets 等字段。"""
    source_id = str(row["source_id"])
    task = row.get("task_type") or row.get("task") or source_tasks.get(source_id, "")
    return {
        "id": str(row["id"]),
        "source_id": source_id,
        "split": row["split"],
        "generator": row["model"],
        "task": task,
    }


def read_metadata(response_file, source_info=None):
    """一次读取身份映射；同文件中的其他字段不会进入返回值。"""
    sources, _ = read_sources(response_file, source_info)
    source_tasks = {}
    for source_id, source in sources.items():
        source_tasks[source_id] = source.get("task_type") or source.get("task", "")

    records = {}
    with Path(response_file).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = metadata_fields(json.loads(line), source_tasks)
            response_id = row["id"]
            if response_id in records:
                raise ValueError(f"重复的 RAGTruth 回答编号：{response_id}")
            records[response_id] = row
    return records


def merge_metadata(identity, response_id, records):
    """按回答编号关联，不覆盖不同的来源或划分，不伪造 token 对齐认证。"""
    row = records.get(str(response_id))
    if row is None:
        return dict(identity)

    merged = dict(identity)
    for key in IDENTITY_KEYS:
        old_value = identity.get(key)
        new_value = row[key]
        if old_value not in (None, "", "unknown") and new_value and old_value != new_value:
            raise ValueError(f"{response_id}: 缓存与原始数据的 {key} 不一致")
        if old_value in (None, "", "unknown") and new_value:
            merged[key] = new_value

    # 编号关联只提供分组身份；offset 和回答校验仍用原 token ID 在评价时验证。
    return merged
