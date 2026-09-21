"""One durable record per model comparison; resume preserves intermediate patch steps."""

from dataclasses import asdict

import numpy as np
from state_audit.experiments.messages import measure_messages
from state_audit.storage import read_arrays, read_json, write_arrays, write_json


class Trials:
    def __init__(self, model, case, groups, directory, progress=None):
        self.model = model
        self.case = case
        self.sites = [site for values in groups.values() for site in values]
        self.directory = directory
        self.records = []
        self.progress = progress

    def evaluate(self, condition, operations=()):
        index = len(self.records)
        path = self.directory / "trials" / f"{index:04d}"
        if path.with_suffix(".json").exists():
            record = read_json(path.with_suffix(".json"))
            if record["condition"] != condition:
                raise ValueError(f"{path}: trial schedule changed; choose a new output")
            arrays = read_arrays(path.with_suffix(".npz"))
            result = record["scores"], unpack_messages(record["messages"], arrays)
        else:
            result = measure_messages(
                self.model, self.case.prefix_ids, self.case.candidates, self.sites, operations
            )
            scores = result[0]
            scores["candidate_sum_margin"] = scores["sum_margin"]
            scores["candidate_mean_margin"] = scores["mean_margin"]
            direction = 1 if self.case.preferred == 0 else -1
            scores["sum_margin"] *= direction
            scores["mean_margin"] *= direction
            arrays, mapping = pack_messages(result[1])
            op_records = pack_operations(operations, arrays)
            write_arrays(path.with_suffix(".npz"), **arrays)
            record = dict(
                condition=condition, scores=result[0], messages=mapping, operations=op_records
            )
            write_json(path.with_suffix(".json"), record)
        self.records.append(dict(index=index, **record))
        if self.progress is not None:
            self.progress.update()
            self.progress.set_postfix_str(condition)
        return result

    def restorer(self, condition):
        """Name each sequential layer patch; inner forwards remain separately inspectable."""
        step = 0

        def evaluate(operations):
            nonlocal step
            result = self.evaluate(f"{condition}/step_{step}", operations)
            step += 1
            return result

        return evaluate


def pack_messages(messages):
    arrays, mapping = {}, {}
    for index, (name, values) in enumerate(messages.items()):
        mapping[name] = {}
        for field, value in values.items():
            key = f"site_{index}_{field}"
            mapping[name][field] = None if value is None else key
            if value is not None:
                arrays[key] = value
    return arrays, mapping


def unpack_messages(mapping, arrays):
    return {
        name: {field: None if key is None else arrays[key] for field, key in fields.items()}
        for name, fields in mapping.items()
    }


def pack_operations(operations, arrays):
    records = []
    for index, operation in enumerate(operations):
        record = dict(operation=type(operation).__name__, **asdict(operation))
        for field in ("value", "weights", "values"):
            if field in record:
                key = f"operation_{index}_{field}"
                arrays[key] = np.asarray(record.pop(field))
                record[field + "_array"] = key
        records.append(record)
    return records
