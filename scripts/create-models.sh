#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

ollama create hammer2.1:7b -f ./modelfiles/hammer2.1_7b.txt

ollama create watt-tool:8b -f ./modelfiles/watt-tool-8b.txt 

ollama create watt-tool:70b -f ./modelfiles/watt-tool-70b.txt 

ollama create functionary-small -f ./modelfiles/functionary-small.txt

ollama create functionary-medium -f ./modelfiles/functionary-medium.txt
