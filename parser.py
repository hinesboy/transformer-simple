import argparse

import torch


parser = argparse.ArgumentParser()

parser.add_argument('--train-file', default='data/train.txt')
parser.add_argument('--dev-file', default='data/dev.txt')

parser.add_argument('--UNK', default=0, type=int)
parser.add_argument('--PAD', default=1, type=int)

# TODO 常改动参数
parser.add_argument('--type', default='train') # 默认是训练模式, 若传递 "evaluate" 则对 dev数据集进行预测输出
parser.add_argument('--gpu', default=3, type=int) # gpu 卡号
parser.add_argument('--epochs', default=5, type=int) # 训练轮数
parser.add_argument('--layers', default=2, type=int) # transformer层数
parser.add_argument('--h-num', default=8, type=int) # multihead attention hidden层数
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--d-model', default=256, type=int) 
parser.add_argument('--d-ff', default=1024, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--max-length', default=60, type=int)
parser.add_argument('--save-file', default='save/model.pt') # 模型保存位置


args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
args.device = device