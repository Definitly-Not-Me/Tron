#!/usr/bin/env bash
python -m src.train "$@" --episodes 20000  --eval-every 100 \
    --eval-episodes 10 --checkpoint-every 500 --seed 3 --render --watch
