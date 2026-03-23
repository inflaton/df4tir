#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

python src/misc/calc_metrics.py results/H100_r2/ H100_r2
python src/misc/calc_metrics.py results/RTX-A6000_r2/ RTX-A6000_r2
python src/misc/calc_metrics.py results/RTX-4090_r2/ RTX-4090_r2
python src/misc/calc_metrics.py results/M3-Max/ M3-Max
python src/misc/calc_metrics.py results/OpenAI/ OpenAI
python src/misc/calc_metrics.py results/Anthropic/ Anthropic
