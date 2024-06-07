#!/bin/bash

# set project path
ROOT="."
FAIRSEQ="${ROOT}/fairseq"
FAIRSCRIPTS="${FAIRSEQ}/scripts"
SPM_TRAIN="${FAIRSCRIPTS}/spm_train.py"
SPM_ENCODE="${FAIRSCRIPTS}/spm_encode.py"
SUBWORD_TYPE="bpe"
DATAROOT="${ROOT}/data/iwslt14"
DATABIN="${ROOT}/data-bin/iwslt14"
mkdir -p $DATABIN


SRCS=(
    "de"
)
TGTS=(
    "en"
)
BPESIZES=(
    5000
    10000
)
PRIME_BPESIZE=${BPESIZES[1]}
TRAIN_MAXLEN=1000


if [ ${#1} -gt 0 ]; then
    echo "Destination Input: $1"
    echo "Setting destination directory to $1 .."
    DEST="${DATABIN}/${1}"
    mkdir -p $DEST/bpe
else
	echo "Destination folder must be set."
	echo "try this USAGE: bash preprocess.sh [dest_folder]"
fi
echo "SRCS = ${SRCS[@]} | TGTS = ${TGTS[@]} | DEST_DIR = "${DEST}" | BPESIZES = ${BPESIZES}" > "${DEST}/data.config.txt"


#####################
#  copy relevant data files from DATAROOT to DATA-BIN
#  'data-bin' is the working directory for experimentr data
#  'data' is the original data-store; don't touch it.
#####################
copy() {

    for SRC in ${SRCS[@]}; do
        for TGT in ${TGTS[@]}; do
            if [ ! -f "${DEST}/train.${SRC}-${TGT}.${TGT}" ]; then
                echo "Copying ${SRC} and ${TGT} data .."
                cp "${DATAROOT}/${SRC}-${TGT}/"*".${SRC}-${TGT}."* "${DEST}"

                echo "Done Copying!"
            else
                echo "Found ${SRC} and ${TGT} data .."
            fi
        done
    done

    for SRC in ${SRCS[@]}; do
        for TGT in ${TGTS[@]}; do
            DIRECTORY="${SRC}-${TGT}"
        done
    done

    if [ ${#SRC[@]} -gt 0 ]; then
        echo "Copying files .."
        for SRC in ${SRCS[@]}; do
            for TGT in ${TGTS[@]}; do
                if [ ! -f "${DEST}/train.${SRC}-${TGT}.${TGT}" ]; then
                    echo "Copying ${SRC} and ${TGT} data .."
                    cp "${DATAROOT}/${DIRECTORY}/"*".${SRC}-${TGT}."* "${DEST}"

                    echo "Done Copying!"
                else
                    echo "Found ${SRC} and ${TGT} data at ${DEST} .."
                fi
            done
        done
    fi

}


#####################
#  learn BPE with sentencepiece
#####################
learn_bpe(){
    TRAIN_FILES=$(
        for SRC in "${SRCS[@]}"; do 
            for TGT in "${TGTS[@]}"; do
                if [ ! ${SRC} = ${TGT} ]; then 
                    echo $DEST/train.${SRC}-${TGT}.${SRC}; echo $DEST/train.${SRC}-${TGT}.${TGT};
                fi
            done 
        done | tr "\n" ",")
    
    echo "learning joint BPE over ${TRAIN_FILES} .."

    for BPESIZE in "${BPESIZES[@]}"; do
        python "${SPM_TRAIN}" \
            --input=${TRAIN_FILES} \
            --model_prefix="${DEST}/bpe/spm.${SUBWORD_TYPE}_${BPESIZE}" \
            --vocab_size=${BPESIZE} \
            --character_coverage=1.0 \
            --model_type="${SUBWORD_TYPE}"
    done
}

######################################
#  Optionally, call functions here
######################################
copy
learn_bpe

######################################
# 5. apply BPE to train/valid/test
######################################

echo "encoding train data with learned BPE..."
for SRC in "${SRCS[@]}"; do
    for TGT in "${TGTS[@]}"; do
        if [ ! ${SRC} = ${TGT} ]; then
            for BPESIZE in "${BPESIZES[@]}"; do
                python "${SPM_ENCODE}" \
                    --model "${DEST}/bpe/spm.${SUBWORD_TYPE}_${BPESIZE}.model" \
                    --output_format=piece \
                    --inputs "${DEST}/train.${SRC}-${TGT}.${SRC}" \
                    --outputs "${DEST}/bpe/train.bpe.${SRC}${BPESIZE}-${TGT}.${SRC}${BPESIZE}" \
                    --min-len 1 --max-len "${TRAIN_MAXLEN}"
            done
            for BPESIZE in "${BPESIZES[@]}"; do
                python "${SPM_ENCODE}" \
                    --model "${DEST}/bpe/spm.${SUBWORD_TYPE}_${PRIME_BPESIZE}.model" \
                    --output_format=piece \
                    --inputs "${DEST}/train.${SRC}-${TGT}.${TGT}" \
                    --outputs "${DEST}/bpe/train.bpe.${SRC}${BPESIZE}-${TGT}.${TGT}" \
                    --min-len 1 --max-len "${TRAIN_MAXLEN}"
            done
        fi
    done
done

echo "encoding valid data with learned BPE..."
for SRC in "${SRCS[@]}"; do
    for TGT in "${TGTS[@]}"; do
        if [ ! ${SRC} = ${TGT} ]; then
            for BPESIZE in "${BPESIZES[@]}"; do
                python "${SPM_ENCODE}" \
                    --model "${DEST}/bpe/spm.${SUBWORD_TYPE}_${BPESIZE}.model" \
                    --output_format=piece \
                    --inputs "${DEST}/valid.${SRC}-${TGT}.${SRC}" \
                    --outputs "${DEST}/bpe/valid.bpe.${SRC}${BPESIZE}-${TGT}.${SRC}${BPESIZE}" \
                    --min-len 1 --max-len "${TRAIN_MAXLEN}"
            done
            for BPESIZE in "${BPESIZES[@]}"; do
                python "${SPM_ENCODE}" \
                    --model "${DEST}/bpe/spm.${SUBWORD_TYPE}_${PRIME_BPESIZE}.model" \
                    --output_format=piece \
                    --inputs "${DEST}/valid.${SRC}-${TGT}.${TGT}" \
                    --outputs "${DEST}/bpe/valid.bpe.${SRC}${BPESIZE}-${TGT}.${TGT}" \
                    --min-len 1 --max-len "${TRAIN_MAXLEN}"
            done
        fi

    done
done

echo "encoding test data with learned BPE ..."
for SRC in "${SRCS[@]}"; do
    for TGT in "${TGTS[@]}"; do
        if [ ! ${SRC} = ${TGT} ]; then
            for BPESIZE in "${BPESIZES[@]}"; do
                python "${SPM_ENCODE}" \
                    --model "${DEST}/bpe/spm.${SUBWORD_TYPE}_${BPESIZE}.model" \
                    --output_format=piece \
                    --inputs "${DEST}/test.${SRC}-${TGT}.${SRC}" \
                    --outputs "${DEST}/bpe/test.bpe.${SRC}${BPESIZE}-${TGT}.${SRC}${BPESIZE}" \
                    --min-len 1 --max-len "${TRAIN_MAXLEN}"
            done
            for BPESIZE in "${BPESIZES[@]}"; do
                python "${SPM_ENCODE}" \
                    --model "${DEST}/bpe/spm.${SUBWORD_TYPE}_${PRIME_BPESIZE}.model" \
                    --output_format=piece \
                    --inputs "${DEST}/test.${SRC}-${TGT}.${TGT}" \
                    --outputs "${DEST}/bpe/test.bpe.${SRC}${BPESIZE}-${TGT}.${TGT}" \
                    --min-len 1 --max-len "${TRAIN_MAXLEN}"
            done
        fi
    done
done

echo ""
echo "Finished encoding train/valid/test with Sentencepiece ${SUBWORD_TYPE} ."

#  usage: bash preprocess.sh [expt_name]
