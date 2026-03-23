#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
BASEDIR=`pwd`

echo Current Directory: $BASEDIR

python src/misc/test_vllm.py --image_url \
    file://$BASEDIR/dataset/attachments/transaction_1.jpeg

python src/misc/test_vllm.py --image_url \
    file://$BASEDIR/dataset/attachments/transaction_999.jpeg