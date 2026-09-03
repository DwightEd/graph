#!/usr/bin/env bash

repository_root="$(realpath "$(dirname "$0")/../..")" || exit $?
PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m experiments.head_resolved_shortcut_route.run all "$@" || exit $?
