"""配置驱动的训练及评估流程、命令行入口。"""
import argparse
import time
from pathlib import Path

# 直接执行本文件时，补上包上下文；包导入及 python -m 启动无需处理。
if __name__ == '__main__' and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'transe_modules'

from .api import TransE
from .config import load_config
from .data import KnowledgeGraphData
from .runtime import device, seed_everything


def run(config):
    """执行完整实验流程：加载数据 → 建模 → 训练与保存 → 加载测试集 → 评估。
    
    config 是 load_config 处理过路径的配置字典，返回两个评估指标。
    """
    # 1. 固定随机种子，并取出数据、模型、训练及输出配置。
    seed_everything(11)
    data_config, model_config = config['data'], config['model']
    training, output = config['training'], config['output']
    # 2. 读取映射与训练/验证三元组，内部列顺序统一为 [h,r,t]。
    data = KnowledgeGraphData.load(data_config['train'], data_config['entity2id'],
                                   data_config['relation2id'], data_config['valid'],
                                   data_config['triple_order'])
    # 3. 将配置和数据交给实验门面；显式传映射，不依赖旧全局字典。
    experiment = TransE(data.entities, data.relations, data.train_triples,
                        embedding_dim=model_config['embedding_dim'],
                        lr=model_config['learning_rate'], margin=model_config['margin'],
                        norm=model_config['norm'], C=model_config['C'],
                        valid_triple_list=data.valid_triples,
                        entity_mapping=data.entities2id, relation_mapping=data.relations2id)
    # 4. 初始化嵌入模型、Bernoulli 采样器和 Adam 优化器。
    experiment.data_initialise()
    # 5. 逐轮训练和验证，按配置路径保存向量、损失记录与曲线。
    experiment.training_run(epochs=training['epochs'], batch_size=training['batch_size'],
                            out_file_title=output['prefix'], num_workers=training.get('num_workers', 0),
                            entity_file=output['entity_embeddings'], relation_file=output['relation_embeddings'])
    # 6. 重新载入保存的向量，并使用同一映射解析测试集。
    experiment.insert_test_data(output['entity_embeddings'], output['relation_embeddings'],
                                data_config['test'], data_config['triple_order'])
    # 7. 执行排名评估；filtered_test=True 时排除其他已知真实答案。
    return experiment.test_run(filter=training.get('filtered_test', False), result_file=output['result_file'])


def main(default_config=None):
    """命令行启动函数，读取 --config 参数并统计整个实验的运行时间。
    
    未传 --config 时使用入口指定的配置或原目录中的 transe_configtiny.json。
    """
    parser = argparse.ArgumentParser(description='Train and evaluate TransE.')
    # 默认配置由脚本位置决定，所以从其他终端目录启动也能找到它。
    parser.add_argument('--config', default=default_config or str(
        Path(__file__).resolve().parent.parent / 'transe_configtiny.json'), help='JSON configuration file')
    args = parser.parse_args()
    start = time.time()
    result = run(load_config(args.config))
    print(f"Total running time: {time.time() - start:.2f} seconds\n{device}")
    return result


if __name__ == '__main__':
    main()
