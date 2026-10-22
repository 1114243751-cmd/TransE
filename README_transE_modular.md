# TransE 模块说明已更新

完整说明见同目录的 `README.md`，包含新的文件结构、命令行参数和运行示例。

现在以根目录 `main.py` 作为训练入口，`evaluate.py` 作为单独评估入口，`metric.py` 负责 MR、MRR、Hits@10。

原 `TransE_pytoch.py` 和 `transe_modules/pipeline.py` 仍可运行，`transe_modules` 中的同名模块会转到根目录的实际实现。
