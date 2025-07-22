jsonfile="datasets/eight.files3.json"
vocabfile="datasets/bert-large-cased-vocab.txt"
prefix="fsi-en-t5-8files-bert-large-cased-vocab-bwplc-small3"

python tools/preprocess_data.py \
               --input $jsonfile \
               --output-prefix $prefix \
               --vocab $vocabfile \
               --dataset-impl mmap \
               --tokenizer-type BertWordPieceCase \
               --split-sentences \
               --workers 8