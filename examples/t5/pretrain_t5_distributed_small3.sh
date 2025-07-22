#!/bin/bash

#GPUS_PER_NODE=8 # use this for real training
GPUS_PER_NODE=1 # only for debug
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0

# TODO not used
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=datasets/fsi-en-t5-8files-bert-large-cased-vocab-bwplc-small3_text_sentence

CHECKPOINT_PATH_IN=/workspace/megatron/ngc_models/release_t5_base
CHECKPOINT_PATH_OUT=/workspace/megatron/ngc_models/release_t5_base

vocabfn=/workspace/megatron/ngc_models/bert-large-cased-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

T5_ARGS="--num-layers 24 \
         --hidden-size 1024 \
         --num-attention-heads 16 \
         --kv-channels 64 \
         --ffn-hidden-size 3072 \
         --encoder-seq-length 512 \
         --decoder-seq-length 128 \
         --max-position-embeddings 512 \
         --lr 0.0001 \
         --lr-decay-style linear \
         --lr-decay-iters 9 \
         --weight-decay 1e-2 \
         --clip-grad 1.0 \
         --train-iters 20 \
         --min-lr 0.00001 \
         --lr-warmup-fraction 0.01 \
         --micro-batch-size 2 \
         --global-batch-size 16 \
         --vocab-file $vocabfn \
         --tokenizer-type BertWordPieceCase \
         --vocab-extra-ids 100 \
         --data-impl mmap \
         --split 949,50,1 \
         --distributed-backend nccl \
         --fp16"

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 5 \
             --eval-interval 1 \
             --eval-iters 1 \
             --activations-checkpoint-method uniform"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       $T5_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH_OUT \
       --load $CHECKPOINT_PATH_IN \
       --data-path $DATA_PATH