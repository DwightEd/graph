#!/usr/bin/env bash

repository_root="$(realpath "$(dirname "$0")/../..")" || exit $?
PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH PYTORCH_CUDA_ALLOC_CONF

python -m experiments.constraint_routing_rhythm.run all "$@" || exit $?
