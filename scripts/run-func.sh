#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
BASEDIR=`pwd`

MODEL=$1
MEM_UTILIZATION=$2

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model-name> <gpu-memory-utilization (default: 0.2)>"
    exit 1
fi

if [ -z "$MEM_UTILIZATION" ]; then
    MEM_UTILIZATION=0.2
fi

echo "GPU Memory Utilization: $MEM_UTILIZATION"
echo Current Directory: $BASEDIR

#vllm serve HuggingFaceTB/SmolVLM-Instruct --dtype auto --api-key ollama \

vllm serve $MODEL --dtype auto --api-key ollama \
    --allowed-local-media-path $BASEDIR/dataset/attachments/ \
    --gpu-memory-utilization $MEM_UTILIZATION \
    --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template ../vllm/examples/tool_chat_template_llama3.1_json.jinja \
    --max_model_len 8192 \
    --quantization bitsandbytes
