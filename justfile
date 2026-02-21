#!/usr/bin/env just --justfile

default:
    @just --list

# ──── Development ────

test *ARGS:
    pytest {{ARGS}}

lint *ARGS:
    ruff check src/ tests/ experiments/ {{ARGS}}

fmt *ARGS:
    ruff format src/ tests/ experiments/ {{ARGS}}

check: lint
    ruff format --check src/ tests/ experiments/

run CONFIG *ARGS:
    python -m experiments.run {{CONFIG}} {{ARGS}}

debug CONFIG *ARGS:
    python -m experiments.run {{CONFIG}} --debug {{ARGS}}

# ──── Remote sync ────

pull-results:
    rsync -avz --partial desktop:~/Code/etd/results/ ./results/

push-datasets:
    rsync -avz --partial ./datasets/ desktop:~/Code/etd/datasets/

push-nuts:
    rsync -avz --partial ./.cache/nuts/ desktop:~/Code/etd/.cache/nuts/

pull-nuts:
    rsync -avz --partial desktop:~/Code/etd/.cache/nuts/ ./.cache/nuts/
