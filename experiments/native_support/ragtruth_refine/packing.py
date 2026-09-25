"""Portable per-task dev/test measurements, without model states or duplicate scores."""

import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from tqdm import tqdm

from .data import read_annotations, read_observed, require_scores


def export_caches(args, reader, manifest):
    require_scores(args.output, manifest)
    archives = []
    for task in args.tasks:
        records = [row for row in manifest["records"] if row["task"] == task]
        truth = read_annotations(reader, manifest, records)
        portable = dict(portable_refinement_cache=True, model=manifest["model"], records=records,
            previous_selected=manifest["previous_selected"], development_sources=manifest["development_sources"],
            selected_answers=len(records), original_scope=manifest["original_scope"])
        path = args.output.with_name(f"{args.output.name}_{task}_cache.zip")
        temporary = path.with_suffix(".partial.zip")
        with ZipFile(temporary, "w", ZIP_DEFLATED) as bundle:
            bundle.writestr("manifest.json", json.dumps(portable, ensure_ascii=False))
            bundle.writestr("annotations.json", json.dumps(truth, ensure_ascii=False))
            bundle.writestr("coverage.json", json.dumps(dict(status="complete")))
            for record in tqdm(records, desc=f"pack {task} dev/test measurements"):
                directory = record["directory"]
                observed = read_observed(reader, record, manifest["previous_selected"])
                buffer = io.BytesIO()
                np.savez(buffer, **observed)
                bundle.writestr(directory + "/observations.npz", buffer.getvalue())
                bundle.writestr(directory + "/response.json", reader.bytes(directory + "/response.json"))
        temporary.replace(path)
        archives.append(str(path))
    return archives
