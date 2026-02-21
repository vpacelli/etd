#!/usr/bin/env just --justfile

default:
    @just --list

pull-results:
    rsync -avz --partial desktop:~/Code/etd/results/ ./results/


push-datasets:
    rsync -avz --partial ./datasets/ desktop:~/Code/etd/datasets/


push-nuts:
    rsync -avz --partial ./.cache/nuts/ desktop:~/Code/etd/.cache/nuts/

pull-nuts:
    rsync -avz --partial desktop:~/Code/etd/.cache/nuts/ ./.cache/nuts/ 
    