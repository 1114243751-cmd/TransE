"""模块化 TransE；新代码建议直接使用对应功能模块。"""
from .api import TransE, dataloader
from .data import KnowledgeGraphData, TripleDataset
from .model import E, TransEModel

__all__ = ['TransE', 'dataloader', 'KnowledgeGraphData', 'TripleDataset', 'E', 'TransEModel']
