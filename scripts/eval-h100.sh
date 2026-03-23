#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

#ollama create functionary-small -f ./modelfiles/functionary-small.txt

#./scripts/eval-model.sh functionary-small

#ollama create functionary-medium -f ./modelfiles/functionary-medium.txt

#./scripts/eval-model.sh functionary-medium
#
./scripts/eval-model.sh cogito:3b

./scripts/eval-model.sh cogito:8b

./scripts/eval-model.sh cogito:14b

./scripts/eval-model.sh cogito:32b

./scripts/eval-model.sh cogito:70b

