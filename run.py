import copy
import os

from parser import args

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_data import PrepareData
from model.attention import MultiHeadedAttention
from model.position_wise_feedforward import PositionwiseFeedForward
from model.embedding import PositionalEncoding, Embeddings
from model.transformer import Transformer
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer
from model.generator import Generator
from lib.criterion import LabelSmoothing
from lib.optimizer import NoamOpt
from train import train
from evaluate import evaluate

def make_model(src_vocab, tgt_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model).to(args.device)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(args.device)
    position = PositionalEncoding(d_model, dropout).to(args.device)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(args.device), N).to(args.device),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout).to(args.device), N).to(args.device),
        nn.Sequential(Embeddings(d_model, src_vocab).to(args.device), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(args.device), c(position)),
        Generator(d_model, tgt_vocab)).to(args.device)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(args.device)

def main():
    # 数据预处理
    data = PrepareData()
    args.src_vocab = len(data.en_word_dict)
    args.tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % args.src_vocab)
    print("tgt_vocab %d" % args.tgt_vocab)

    # 初始化模型
    model = make_model(
                        args.src_vocab, 
                        args.tgt_vocab, 
                        args.layers, 
                        args.d_model, 
                        args.d_ff,
                        args.h_num,
                        args.dropout
                    )

   
    if args.type == 'train':
        # 训练
        print(">>>>>>> start train")
        criterion = LabelSmoothing(args.tgt_vocab, padding_idx = 0, smoothing= 0.0)
        optimizer = NoamOpt(args.d_model, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))
        
        train(data, model, criterion, optimizer)
        print("<<<<<<< finished train")
    elif args.type == "evaluate":
        # 预测
        # 先判断模型有没有训练好(前提)
        if os.path.exists(args.save_file):
            # 加载模型
            model.load_state_dict(torch.load(args.save_file))
            # 开始预测
            print(">>>>>>> start evaluate")
            evaluate(data, model)         
            print("<<<<<<< finished evaluate")
        else:
            print("Error: pleas train before evaluate")
    else:
        print("Error: please select type within [train / evaluate]")

if __name__ == "__main__":
    main()