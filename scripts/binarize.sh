#!/bin/bash

# set project path
ROOT="."
FAIRSEQ="${ROOT}/fairseq"
FAIRSCRIPTS="${FAIRSEQ}/scripts"
DATAROOT="${ROOT}/data/iwslt14"
DATABIN="${ROOT}/data-bin/iwslt14"
mkdir -p $DATABIN


SRCS=(
    "de5000"
    "de10000"
)
TGTS=(
    "en"
)
BPESIZES=(
    5000
    10000
)
MAX_BPESIZE="${BPESIZES[1]}"
TEXT="${DATABIN}/de-en_5k_10k"
mkdir -p "${TEXT}/bin"
echo "make sure this is the config you want:"
cat "${TEXT}/data.config.txt"
echo ""
##############################

for BPESIZE in "${BPESIZES[@]}"; do

    DICT="jointdict_${BPESIZE}.txt"
    echo "Generating joined dictionary for all languages based on BPE.."
    # strip the first three special tokens and append fake counts for each vocabulary
    tail -n +4 "${TEXT}/bpe/spm.bpe_${BPESIZE}.vocab" | cut -f1 | sed 's/$/ 100/g' > "${TEXT}/bin/${DICT}"

done

echo "binarizing pairwise langs .."
MAX_DICT="jointdict_${MAX_BPESIZE}.txt"
for i in ${!BPESIZES[@]}; do

    BPESIZE=${BPESIZES[i]}
    SRC=${SRCS[i]}
    TGT=${TGTS[0]}

    echo "binarizing data ${SRC}-${TGT} data.."
    fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
        --destdir "${TEXT}/bin" \
        --trainpref "${TEXT}/bpe/train.bpe.${SRC}-${TGT}" \
        --validpref "${TEXT}/bpe/valid.bpe.${SRC}-${TGT}" \
        --testpref "${TEXT}/bpe/test.bpe.${SRC}-${TGT}" \
        --srcdict "${TEXT}/bin/${MAX_DICT}" --tgtdict "${TEXT}/bin/${MAX_DICT}" \
        --workers 4
done

echo ""
echo "Creating langs file based on binarised dicts .."
{
    for SRC in "${SRCS[@]}"; do
        echo "$SRC"
    done
    for TGT in "${TGTS[@]}"; do
        echo "$TGT"
    done
} > ${TEXT}/bin/langs.file
echo "--> ${TEXT}/bin/langs.file"

#  usage: bash binarize.sh
