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