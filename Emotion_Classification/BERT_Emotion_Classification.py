"""
    思路：
    1、huggingface按照BERT预训练微调的方式进行训练和预测(利用字段：文本内容和标签)
    2、huggingface按照BERT预训练微调的方式进行训练和预测(利用全字段)
    对比1和2的效果
"""
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_process.data_process import get_csv_data
from EM_DataSet import Emotion_Dataset
from transformers import BertTokenizer
from transformers import BertModel
from net import EM_CLS_Net
from transformers import AdamW
from Trainning.trainner import Trainner

# DEVICE
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
# Bert Path
BERT_PATH = 'pretrained_model/bert-base-chinese'

def collate_fn(data):
    """
        torch.Dataset转torch.Dataloader时需要的数据批处理方法
        按照collate_fn返回的值形成dataloader,这样应该也可以省去原来的zip部分
    """
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 用基础的bert-base-chinese
    token = BertTokenizer.from_pretrained(BERT_PATH)
    # 编码看下参数意义
    """
        编码的五种方式(都是将text生成为id)：https://zhuanlan.zhihu.com/p/424565138
        tokenizer.encode(text)
        tokenizer.encode_plus(text)
        tokenizer.batch_encode_plus([text])
        tokenizer(text)
        tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(text)
        )
    """
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,# 按批处理句子
                                   truncation=True,# 按最大长度截断句子
                                   padding='max_length',# 按最大长度padding
                                   max_length=64,# 设置最大长度,不设置默认是512,超过512会截断
                                   return_tensors='pt',# 默认是None,'tf'表示tensorflow版本,'pt'表示pytorch版本
                                   return_length=True)
    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.Tensor(labels)

    return input_ids, attention_mask, token_type_ids, labels  


def main():
    """

    """
    train_data, valid_data = get_csv_data(file_path='data_process/dataset/Automobile_UserPoint_train.csv')
    emotion_train_ds = Emotion_Dataset(train_data)
    emotion_valid_ds = Emotion_Dataset(valid_data)
    # 数据加载器
    emotion_train_dl = DataLoader(dataset=emotion_train_ds,
                                  batch_size=64,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=True)
    emotion_valid_dl = DataLoader(dataset=emotion_valid_ds,
                                  batch_size=64,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=True)
    # 加载预训练模型&下游任务模型
    pretrained = BertModel.from_pretrained(BERT_PATH)
    # 不训练,不需要计算梯度
    for param in pretrained.parameters():
        param.requires_grad_(False)
    # 模型
    EM_model = EM_CLS_Net(pretrained_model=pretrained)
    optimizer = AdamW(EM_model.parameters(), lr=5e-4, weight_decay=0.01)
    # trainner训练器
    trainner =  Trainner()
    trainner.training(
                        model = EM_model,
                        device = DEVICE,
                        epochs = 6,
                        train_dl = emotion_train_dl,
                        valid_dl = emotion_valid_dl,
                        criterion = nn.CrossEntropyLoss(),
                        optimizer = optimizer
                    )


if __name__ == '__main__':
        main()