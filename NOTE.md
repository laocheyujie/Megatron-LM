## 数据
```bash
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
```

- output:
    1. .bin
    2. .idx
- mmap: 数据保存的时候，是 memory-map file

### args
```py
Namespace(
    append_eod=False, 
    dataset_impl='mmap', 
    emoji_file='C:\\Users\\user\\source\\repos\\gpt2-japanese\\emoji.json', 
    input='datasets/english-fsi/eight.files3.json', 
    json_keys=['text'], 
    keep_empty=False, 
    keep_newlines=False, 
    log_interval=100, 
    make_vocab_size_divisible_by=128, 
    mecab_dict_path=None, 
    merge_file=None, 
    output_prefix='fsi-en-bert-8files-bert-large-cased-vocab-bwplc-small3', 
    rank=0, 
    split_sentences=True, 
    tensor_model_parallel_size=1, 
    tokenizer_type='BertWordPieceCase', 
    vocab_file='datasets/bert-large-cased-vocab.txt', 
    workers=1
)
```

### .idx 文件的格式（二进制）
|NOTE no.| length (by bytes) | name | value (examples) |
|--------|--------|--------|--------|
|1 | 9 bytes - 固定的  | cls._HDR_MAGIC  | b'MMIDIDX\x00\x00'  |
|2 | 8 bytes - 固定的  | struct.pack('<Q', 1)  | b'\x01\x00\x00\x00\x00\x00\x00\x00'  |
|3 | 1 byte - 固定的  | struct.pack('<B', code(dtype))  | b'\x08'  |
|4 | 8 bytes - 长度是固定的</br>具体的值是 corpus(.json文件中) 所有句子的数量总和</br>（两个 .idx 文件合并时直接相加即可）  | struct.pack('<Q', len(sizes))  | b'!\x19\x00\x00\x00\x00\x00\x00' (=6433)，即，所有文档中句子的个数的总和，而 sizes 中保存的是每个句子中 word_piece/token 的数量，类似于：sizes[0:5]  [26, 264, 24, 56, 24] （独立的效果）  |
|5 | 8 bytes - 长度是固定的</br>其取值是 .json 文件中所有文档的数量 +1，（两个 .idx 文件合并的时候，这两个值相加，然后 -1 即可）  | struct.pack('<Q', len(doc_idx))  | b'\xe9\x03\x00\x00\x00\x00\x00\x00' (=1001)，即，json 文件中的文档的个数 +1，因为最初的一个是 0，例如：doc_idx[0:10]  [0, 2, 8, 16, 24, 30, 36, 42, 48, 56]；这里，2 代表的是最初一个文档里面有 2 个句子，8 代表的是前两个文档中有 8 个句子，即第二个文档中是 6 个句子，以此类推（累加效果）  |
|6 | (4 * seq_len) bytes - 动态的</br>根据句子的个数而定（整个 corpus 中的句子的个数）（两个.idx文件合并的时候，两个sizes数组直接拼接起来即可）  | sizes.tobytes(order='C')  | 把 sizes 这个数组，保存到 idx 文件，因为数组有 seq_len 个句子，每个元素保存的是一个句子中 word_piece/token 的数量，所以要保存这个数组，需要的 bytes 的数量就是 (4 * seq_len) bytes |
|7 | (8 * seq_len) bytes - 动态的</br>根据整个corpus中的句子的个数而定，这其中因为 pointers 中的值被转换为了 np.int64，是 8 个 bytes 来表示一个 pointer 的值了（两个 .idx 文件合并的时候，需要把第二个 pointers 数组的所有值加上：sum(sizes) * pointers[1] / sizes[0]）  | pointers.tobyptes(order='C')  | pointers 是基于 sizes 计算得到的每个句子的“相对内存地址（的起始位置）”，例如：pointers[0:5]  [0, 52, 580, 628, 740] 因为 sizes[0] = 26，而且 dtype().itemsize = 2，则 0 之后的 pointer 的值为 26\*2=52；再往后，sizes[1] = 264，则 pointer 的值为：264\*2+52=580，以此类推  |
|8 | (8 * (doc_num + 1)) bytes - 动态的</br>根据 corpus 中的文档的个数而定（两个 .idx 文件合并的时候，需要把第二个 doc_idx 数组的（1）第一个元素 0 删除，（2）剩余每个元素都加上第一个 doc_idx 数组的最后一个元素）例如，[0, 2, 8] 和 [0, 2, 8] 合并的时候，得到的 doc_idx 应该是：[0, 2, 8, 10, 16] | doc_idx.tobytes(order='C')  | 累加的每个文档中的句子的个数，doc_idx[0] = 0，这里是转换为 np.int64 进行存储。例如：doc_idx[-10:]  [6379, 6385, 6391, 6397, 6407, 6413, 6415, 6425, 6427, 6433]  |



## 构造

1. 获取模型：`model_provider`返回模型普通版本（vanilla version）的 cpu 模型，没有 fp16 或 ddp，但是已经被 Megatron 改造为并行的版本
2. 获取数据集：`train_valid_test_datasets_provider`接收 train/valid/test 数据集的大小，返回 train, valid, test 数据集
3. 前向传播函数：`forward_step`接收“数据迭代器”和“模型”，返回`loss`标量，该标量带有一个字典，其中 key:value 是希望在训练期间监视信息，例如`lm loss:value`
    - 广播数据：`forward_step`会调用`get_batch`获取`batch`数据，其内部会从数据迭代器获取数据，然后使用`broadcast_data`函数，把输入数据从`rank 0`广播到所有`tensor-model-parallel`的其他 ranks 上


## Pretrain

1. 初始化 Megatron
2. 使用`model_provider`设置模型、优化器和 lr 计划
3. 调用`train_val_test_data_provider`获取 train/val/test 数据集
4. 使用`forward_step_func`训练模型


## _initialize_distributed

位于：megatron/training/initialize.py:_initialize_distributed

1. 调用`torch.distributed.init_process_group`初始化分布式环境
    - `torch.distributed.init_process_group`会生成一个进程组，同组内进程训练同一个模型，也能确定用什么方式进行通信
    - 进程组会给组内每个进程一个序号，就是`gloabl_rank`，如果是多机并行，每个机器内部创建的进程之间也有一个序号，就是`local_rank`；如果是单机多卡并行，`local_rank`和`global_rank`是一致的
2. 调用`mpu.initialize_model_parallel`设置模型并行，数据并行等各种进程组，每个`rank`对应进程都有自己全局变量
    - `_TENSOR_MODEL_PARALLEL_GROUP`：当前 rank 所属的 Intra-layer model parallel group，`TP`进程组 
    - `_PIPELINE_MODEL_PARALLEL_GROUP`：当前 rank 所属的 Intra-layer model parallel group，`PP`进程组 
    - `_MODEL_PARALLEL_GROUP`：当前 rank 所属于`MP`进程组，包括了`TP`和`PP` 
    - `_EMBEDDING_GROUP`： Embedding 并行对应进程组
    - `_CONTEXT_PARALLEL_GROUP`: 当前 rank 所属的`CP`进程组
    - `_EXPERT_MODEL_PARALLEL_GROUP`: 当前 rank 所属的`EP`进程组
    - `_DATA_PARALLEL_GROUP`：当前 rank 所属的`DP`进程组



## References

- https://zhuanlan.zhihu.com/p/388830967