#!/bin/bash

# set project path
ROOT="."
FAIRSEQ="${ROOT}/fairseq"
DATAROOT="${ROOT}/data/iwslt14"
DATABIN="${ROOT}/data-bin/iwslt14"

train() {
    DIR="de-en_5k_10k"
    DATA="${DATABIN}/${DIR}/bin"

    LANG_LIST="${DATA}/langs.file"
    LANG_PAIRS="de5000-en,de10000-en"
    EVAL_LANG_PAIRS="de10000-en,"

    SAMP_MAIN='"main:de10000-en":0.0'
    SAMP_TGT='"main:de5000-en":1.0'
    SAMPLE_WEIGHTS="{${SAMP_TGT},${SAMP_MAIN}} --virtual-data-size 641000"
    EXPTNAME="iwslt14_de-en_5k_10k"
    RUN="#0"

    CKPT="checkpoints/${EXPTNAME}"
    EXPTDIR="experiments/${EXPTNAME}"
    mkdir -p "$EXPTDIR"
    mkdir -p "$CKPT"

    TASK="translation_multi_simple_epoch_drda --prime-src de10000 --prime-tgt en" # !
    LOSS="label_smoothed_cross_entropy_js --js-alpha 5 --js-warmup 2000"
    ARCH="transformer_iwslt_de_en"
    MAX_EPOCH=200
    PATIENCE=25
    MAX_TOK=5000
    UPDATE_FREQ=4
    LR=6e-4
    DROP=0.3
    WARMUP=8000
    DEVICES=0
    WANDB="iwslt14_de-en"
}

######## exp config call #########
train

######## training begins here ####

echo "${EXPTNAME}"
echo "${ARCH}"
echo "${LOSS}"

echo "Entering training.."

export CUDA_VISIBLE_DEVICES=${DEVICES}
python  ${FAIRSEQ}/train.py --log-format simple --log-interval 200 ${TEST} \
    $DATA --save-dir ${CKPT} --keep-best-checkpoints 2 \
    --fp16 --fp16-init-scale 64 --empty-cache-freq 200 \
    --lang-dict "${LANG_LIST}" --lang-pairs "${LANG_PAIRS}" \
    --task ${TASK} \
    --arch ${ARCH} --share-all-embeddings \
    --sampling-weights ${SAMPLE_WEIGHTS} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr ${LR} --lr-scheduler inverse_sqrt --warmup-updates ${WARMUP} \
    --dropout ${DROP} --attention-dropout 0.1 --weight-decay 0.0001 \
    --patience ${PATIENCE} \
    --keep-last-epochs 6 \
    --criterion ${LOSS} \
    --label-smoothing 0.1 \
    --ddp-backend legacy_ddp --num-workers 0 \
    --max-tokens ${MAX_TOK} --update-freq ${UPDATE_FREQ} --eval-bleu \
    --eval-lang-pairs ${EVAL_LANG_PAIRS} --validate-after-updates 1000 \
    --valid-subset valid --ignore-unused-valid-subsets --batch-size-valid 200 \
    --eval-bleu-detok moses --eval-bleu-remove-bpe sentencepiece \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
    --wandb-project ${WANDB} | tee -a "${EXPTDIR}/train.${RUN}.log"


#  usage: bash train.sh