#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

ollama pull qwen2.5:3b

./scripts/eval-model.sh qwen2.5:3b

ollama pull qwen2.5:7b

./scripts/eval-model.sh qwen2.5:7b

#ollama pull llama3.1:8b

#./scripts/eval-model.sh llama3.1:8b

#ollama create functionary-small -f ./modelfiles/functionary-small.txt

#./scripts/eval-model.sh functionary-small

ollama pull qwen2.5:14b

./scripts/eval-model.sh qwen2.5:14b

ollama pull qwen2.5:32b

./scripts/eval-model.sh qwen2.5:32b

#ollama pull llama3.1:70b

#./scripts/eval-model.sh llama3.1:70b

#ollama create functionary-medium -f ./modelfiles/functionary-medium.txt

#./scripts/eval-model.sh functionary-medium

ollama pull qwen2.5:72b

./scripts/eval-model.sh qwen2.5:72b

