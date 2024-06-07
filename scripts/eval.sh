#!/bin/bash

# set project path
ROOT="."
FAIRSEQ="${ROOT}/fairseq"
FAIRSCRIPTS="${FAIRSEQ}/scripts"
DATAROOT="${ROOT}/data"
DATABIN="${ROOT}/data-bin"

EXPTSUF=$1
GEN=$2
BPESIZE=$3

SRC="de"
TGT="en"


iwslt() {
    EXPTNAME="iwslt14_${EXPTSUF}"
    DATA="${DATABIN}/iwslt14/${EXPTSUF}"
    BEAM=5
    LENPEN=1.0
    CKPTDIR="checkpoints/${EXPTNAME}"
    EXPTDIR="experiments/${EXPTNAME}"
    LANG_PAIRS="${SRC}-${TGT}"
    LANG_LIST="${DATA}/bin/langs.file"
}
wmt() {
    EXPTNAME="wmt14_${EXPTSUF}"
    DATA="${DATABIN}/wmt14/${EXPTSUF}"
    BEAM=4
    LENPEN=0.6
    CKPTDIR="checkpoints/${EXPTNAME}"
    EXPTDIR="experiments/${EXPTNAME}"
    LANG_PAIRS="${SRC}-${TGT}"
    LANG_LIST="${DATA}/bin/langs.file"
}

interactive() {

    echo "sanity -- src and tgt wc -l"
    wc -l ${DATA}/${GEN}.${SRC}-${TGT}.${SRC}
    wc -l ${DATA}/${GEN}.${SRC}-${TGT}.${TGT}

    python "${FAIRSEQ}/fairseq_cli/interactive.py" "${DATA}/bin" \
        --input="${DATA}/${GEN}.${SRC}-${TGT}.${SRC}" \
        --bpe sentencepiece --sentencepiece-model "${DATA}/bpe/spm.bpe_${BPESIZE}.model" \
        --source-lang "${SRC}${BPESIZE}" --target-lang ${TGT} \
        --task translation_multi_simple_epoch \
        --lang-tok-style "multilingual" \
        --path ${CKPTDIR}/${CKPT} \
        --buffer-size 300  --batch-size 296 \
        --beam ${BEAM} --lenpen ${LENPEN} --remove-bpe=sentencepiece  \
        --lang-dict "${LANG_LIST}" \
        --lang-pairs "${LANG_PAIRS}"
}

gen() {

    RES="${EXPTDIR}/gen.avg${AVG}"
    mkdir -p "${RES}/bpe${BPESIZE}"

    interactive | tee "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.out"

    # extract hypotheses
    grep ^H "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.out" | cut -f3- > "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.sys"

    # prepare refs
    cat "${DATA}/${GEN}.${SRC}-${TGT}.${TGT}" > "${RES}/bpe${BPESIZE}/${GEN}.${SRC}-${TGT}.${TGT}.ref"

    SYS="${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.sys"
    REF="${RES}/bpe${BPESIZE}/${GEN}.${SRC}-${TGT}.${TGT}.ref"

    ########### compute SacreBLEU ###########
    sacrebleu "${RES}/bpe${BPESIZE}/${GEN}.${SRC}-${TGT}.${TGT}.ref" -i "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.sys" -m bleu -w 4 \
    -l ${SRC}-${TGT} \
    > "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.sacrebleu"

    ########### compute CompoundBLEU ###########
    sacremoses -j 4 tokenize < $SYS | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > ${SYS}.com
    sacremoses -j 4 tokenize < $REF | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > ${REF}.com
    fairseq-score -r ${REF}.com -s ${SYS}.com > "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.combleu" 

    ########### compute multibleu like previous work ############
    sacremoses -j 4 tokenize < "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.sys" > "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.sys.multi"
    sacremoses -j 4 tokenize < "${RES}/bpe${BPESIZE}/${GEN}.${SRC}-${TGT}.${TGT}.ref" > "${RES}/bpe${BPESIZE}/${GEN}.${SRC}-${TGT}.${TGT}.ref.multi"
    fairseq-score -r "${RES}/bpe${BPESIZE}/${GEN}.${SRC}-${TGT}.${TGT}.ref.multi" -s "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.sys.multi" \
    > "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.multibleu"

    ################ display scores #############
    echo "sacreBLEU"
    cat "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.sacrebleu"
    echo ""
    echo "compoundBLEU"
    cat "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.combleu"
    echo ""
    echo "multiBLEU"
    cat "${RES}/bpe${BPESIZE}/gen.${GEN}.${SRC}-${TGT}.${TGT}.multibleu"
}


export CUDA_VISIBLE_DEVICES=0
iwslt

AVG=5
python ${FAIRSEQ}/scripts/average_checkpoints.py \
    --inputs ${CKPTDIR} \
    --output "${ROOT}/${CKPTDIR}/checkpoint.avg${AVG}.pt" \
    --num-epoch-checkpoints ${AVG}
CKPT="checkpoint.avg${AVG}.pt"
gen

#  usage: bash eval.sh de-en_5k_10k test 10000
