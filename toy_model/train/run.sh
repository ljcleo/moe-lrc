#!/usr/bin/env bash

cfg_exp=${1}
cfg_device=${2}

cfg_h_dim=1280
cfg_total_f_dim=40960

cfg_eval_iter=500
cfg_warmup_iter=${cfg_eval_iter}
cfg_save_iter=$((${cfg_eval_iter} * 2))
cfg_train_iter=$((${cfg_eval_iter} * 20))

case ${cfg_exp} in
    baseline)
        cfg_total_n_experts=64
        cfg_total_top_k=8
        cfg_n_shared=0
        cfg_moe_layers=1
        cfg_dense_slim=0
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=1e-3
        ;;
    32expert)
        cfg_total_n_experts=32
        cfg_total_top_k=4
        cfg_n_shared=0
        cfg_moe_layers=1
        cfg_dense_slim=0
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=1e-3
        ;;
    top16)
        cfg_total_n_experts=64
        cfg_total_top_k=16
        cfg_n_shared=0
        cfg_moe_layers=1
        cfg_dense_slim=0
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=1e-3
        ;;
    top2)
        cfg_total_n_experts=64
        cfg_total_top_k=2
        cfg_n_shared=0
        cfg_moe_layers=1
        cfg_dense_slim=0
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=1e-3
        ;;
    share1)
        cfg_total_n_experts=64
        cfg_total_top_k=8
        cfg_n_shared=1
        cfg_moe_layers=1
        cfg_dense_slim=0
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=1e-3
        ;;
    share2)
        cfg_total_n_experts=64
        cfg_total_top_k=8
        cfg_n_shared=2
        cfg_moe_layers=1
        cfg_dense_slim=0
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=1e-3
        ;;
    skip1)
        cfg_total_n_experts=64
        cfg_total_top_k=8
        cfg_n_shared=0
        cfg_moe_layers="[0]+[1]*7"
        cfg_dense_slim=0
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=1e-3
        ;;
    sparse2)
        cfg_total_n_experts=64
        cfg_total_top_k=8
        cfg_n_shared=0
        cfg_moe_layers="[0,1]*4"
        cfg_dense_slim=0
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=1e-3
        ;;
    skip1slim)
        cfg_total_n_experts=64
        cfg_total_top_k=8
        cfg_n_shared=0
        cfg_moe_layers="[0]+[1]*7"
        cfg_dense_slim=1
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=1e-3
        ;;
    sparse2slim)
        cfg_total_n_experts=64
        cfg_total_top_k=8
        cfg_n_shared=0
        cfg_moe_layers="[0,1]*4"
        cfg_dense_slim=1
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=1e-3
        ;;
    nolb)
        cfg_total_n_experts=64
        cfg_total_top_k=8
        cfg_n_shared=0
        cfg_moe_layers=1
        cfg_dense_slim=0
        cfg_aux_loss_coeff=0
        cfg_z_loss_coeff=1e-3
        ;;
    overlb)
        cfg_total_n_experts=64
        cfg_total_top_k=8
        cfg_n_shared=0
        cfg_moe_layers=1
        cfg_dense_slim=0
        cfg_aux_loss_coeff=1e-1
        cfg_z_loss_coeff=1e-3
        ;;
    nozl)
        cfg_total_n_experts=64
        cfg_total_top_k=8
        cfg_n_shared=0
        cfg_moe_layers=1
        cfg_dense_slim=0
        cfg_aux_loss_coeff=1e-2
        cfg_z_loss_coeff=0
        ;;
    *)
        echo "Invalid experiment: ${cfg_exp}"
        exit 1
        ;;
esac

cfg_n_experts=$((${cfg_total_n_experts} - ${cfg_n_shared}))
cfg_top_k=$((${cfg_total_top_k} - ${cfg_n_shared}))
cfg_f_dim=$((${cfg_total_f_dim} / ${cfg_total_n_experts}))

if [[ "${cfg_dense_slim}" -eq 1 ]]; then
    cfg_n_dense_rep=$((${cfg_top_k} + ${cfg_n_shared}))
else
    cfg_n_dense_rep=$((${cfg_n_experts} + ${cfg_n_shared}))
fi

export CUDA_VISIBLE_DEVICES=${cfg_device}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=2
export WANDB_MODE="offline"

GPUS_PER_NODE=$(nvidia-smi --query-gpu name -i ${cfg_device} --format csv,noheader | wc -l)
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"600${cfg_device:0:1}"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$((${GPUS_PER_NODE} * ${NNODES}))

ROOT_PATH=/root/moe-lrc
TOK_PATH="${ROOT_PATH}/tokenizer"
DAT_PATH="${ROOT_PATH}/data${cfg_device:0:1}/train_text_document"
EXP_PATH="${ROOT_PATH}/experiments/${cfg_exp}"
mkdir -p "${EXP_PATH}"

export WANDB_DATA_DIR="${EXP_PATH}/wandb"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 8
    --hidden-size ${cfg_h_dim}
    --ffn-hidden-size $((${cfg_f_dim} * ${cfg_n_dense_rep}))
    --num-attention-heads 10
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --qk-layernorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 5
    --position-embedding-type rope
    --rotary-base 10000
    --disable-bias-linear
    --seed 19260817
)

MOE_ARGS=(
    --moe-layer-freq ${cfg_moe_layers}
    --moe-ffn-hidden-size ${cfg_f_dim}
    --num-experts ${cfg_n_experts}
    --moe-router-topk ${cfg_top_k}
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff ${cfg_aux_loss_coeff}
    --moe-z-loss-coeff ${cfg_z_loss_coeff}
    --moe-grouped-gemm
    --moe-token-dispatcher-type allgather
    --moe-router-dtype fp32
)

if [[ ${cfg_n_shared} -ge 1 ]]; then
    MOE_ARGS+=("--moe-shared-expert-intermediate-size $((${cfg_f_dim} * ${cfg_n_shared}))")
fi

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOK_PATH}"
    --legacy-tokenizer
    --data-path "${DAT_PATH}"
    --split 99998,1,1
)

TRAINING_ARGS=(
    --micro-batch-size 16
    --global-batch-size 1024
    --lr 4e-4
    --train-iters ${cfg_train_iter}
    --lr-decay-iters ${cfg_train_iter}
    --lr-decay-style cosine
    --min-lr 5e-5
    --weight-decay 0.1
    --lr-warmup-iters ${cfg_warmup_iter}
    --clip-grad 1.0
    --bf16
    --overlap-grad-reduce
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval ${cfg_save_iter}
    --eval-interval ${cfg_eval_iter}
    --eval-iters 1
    --save "${EXP_PATH}"
    --ckpt-format torch
    --tensorboard-dir "${EXP_PATH}/tensorboard"
    --log-validation-ppl-to-tensorboard
    --wandb-project moe-lrc
    --wandb-exp-name ${cfg_exp}
    --wandb-save-dir "${WANDB_DATA_DIR}"
)

CMD="torchrun ${DISTRIBUTED_ARGS[@]} Megatron-LM/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}"

echo ${CMD}

cd $(dirname ${0})
source .venv/bin/activate
${CMD} | tee ${EXP_PATH}/log.txt
deactivate
rsync -a --progress ${EXP_PATH} ./experiments/
