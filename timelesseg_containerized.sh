#!/usr/bin/env bash

export CONFIG_YAML="/app/config.yaml"

cd /app && \
    python3 entrypoint.py "$@"