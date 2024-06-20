#!/usr/bin/env bash

# Noise multpliers 0.0,8.7,3.4,1.12,0.845,0.567,0.3543

export $(cat .env | xargs) && python3 causal_inference/simulation_cli.py $1 $2 $3 $4 $5
