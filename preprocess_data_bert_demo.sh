jsonfile="datasets/english-fsi/eight.files3.json"
vocabfile="datasets/bert-large-cased-vocab.txt"
prefix="fsi-en-bert-8files-bert-large-cased-vocab-bwplc-small3"

python tools/preprocess_data.py \
        --input $jsonfile \
        --output-prefix $prefix \
        --vocab $vocabfile \
        --dataset-impl mmap \
        --tokenizer-type BertWordPieceCase
        # --tokenizer-type BertWordPieceLowerCase
        # --split-sentences 
