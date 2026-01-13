#!/bin/bash

# Setting hyperparameters
ALGORITHM=${1:-"dps-fd"}

OUTPUT_DIR="/gemini/code/saves"

# heterogeneous
CENTRAL_MODEL="/gemini/pretrain/server_roberta-large"
LOCAL_MODELS="/gemini/code/model/bert-base-cased,/gemini/code/model/bert-large-cased,/gemini/code/model/roberta-base,/gemini/code/model/roberta-large,/gemini/code/model/xlnet-large-cased,/gemini/code/model/bert-base-cased,/gemini/code/model/bert-base-cased,/gemini/code/model/roberta-base,/gemini/code/model/roberta-base,/gemini/code/model/bert-base-cased,"

# homogeneous
# CENTRAL_MODEL="/gemini/code/model/bert-base-cased"
# LOCAL_MODELS="/gemini/code/model/bert-base-cased,/gemini/code/model/bert-base-cased,/gemini/code/model/bert-base-cased,/gemini/code/model/bert-base-cased,/gemini/code/model/bert-base-cased"

R=${2:-5}
DATA_DIR=${3:-"/gemini/data-1/all_domain_test"}
CLIENT_NUM=${4:-1}
EPOCHS=3
DIS_EPOCHS=3
LR=2e-5
BATCH_SIZE=32
MAX_SEQ_LEN=128
WEIGHT_DECAY=0.01
PUBLIC_RATIO=0.2

TOP_LOCAL_SAMPLE=${5:-25000}
TOP_GLOBAL_SAMPLE=${6:-35000}

DO_TEST="--do_test"

# Run
python /gemini/code/main.py \
    --algorithm $ALGORITHM \
    --output_dir $OUTPUT_DIR \
    --local_models $LOCAL_MODELS \
    --central_model $CENTRAL_MODEL \
    --K $CLIENT_NUM \
    --R $R \
    --data_dir $DATA_DIR \
    --E $EPOCHS \
    --dis_epochs $DIS_EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --weight_decay $WEIGHT_DECAY \
    --max_seq_length $MAX_SEQ_LEN \
    --public_ratio $PUBLIC_RATIO \
    --top_local_sample $TOP_LOCAL_SAMPLE \
    --top_global_sample $TOP_GLOBAL_SAMPLE \
    $DO_TEST