#!/usr/bin/env bash

python -m experiments.reanchor_flow.run corridor "$@" || exit $?
