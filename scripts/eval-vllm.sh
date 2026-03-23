#!/bin/sh

BASEDIR=$(dirname $0)
cd $BASEDIR/..
echo Current Directory:
pwd

export MODEL=$1
MEM_UTILIZATION=$2
SLEEP_TIME=$3

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model-name> <gpu-memory-utilization (default: 0.4)> <sleep-time-in-seconds (default: 300)>"
    exit 1
fi

if [ -z "$MEM_UTILIZATION" ]; then
    MEM_UTILIZATION=0.4
fi

if [ -z "$SLEEP_TIME" ]; then
    SLEEP_TIME=300
fi

export OPENAI_API_KEY=ollama
export BASE_URL=http://localhost:8000/v1
export FINANCE_CLERK_MODEL=$MODEL
export SUPERVISOR_MODEL=$MODEL
export SQL_MODEL=$MODEL

#export VISION_BASE_URL=http://localhost:8000/v1
#export VISION_MODEL=$MODEL

# Start the script in the background
# ./scripts/run-func.sh $MODEL $MEM_UTILIZATION > vllm.log 2>&1 &
vllm serve $MODEL --dtype auto --api-key ollama \
    --gpu-memory-utilization $MEM_UTILIZATION \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max_model_len 8192 \
    --quantization bitsandbytes 2>&1 > vllm.log &
script_pid=$!

# Main script operations
echo "Waiting for vLLM server to start up ..."
sleep $SLEEP_TIME
echo "vLLM server started with PID: $script_pid"

echo Evaluating $MODEL
python app.py

kill $script_pid
# Wait for the background process to finish
wait $script_pid
