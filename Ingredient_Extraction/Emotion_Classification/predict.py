"""
    执行命令：
    python predict.py --predict_one --model_path output/two_classes_checkpoints/2022-03-09_model_state.pt 
"""
import argparse
import warnings
import numpy as np
import torch
from data_process.data_process import get_two_classes_csv_data
from torch.utils.data import DataLoader
from net import EM_CLS_Net
from EM_DataSet import Emotion_Dataset
from transformers import BertTokenizer
from transformers import BertModel
from utils.logging_util import get_logging
warnings.filterwarnings("ignore")

# DEVICE
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
# Bert Path
BERT_PATH = 'pretrained_model/bert-base-chinese'
# logging
LOG_DIR = 'output/predict_log/'
logging = get_logging(LOG_DIR=LOG_DIR)

def collate_fn(data):
    """
        torch.Dataset转torch.Dataloader时需要的数据批处理方法
        按照collate_fn返回的值形成dataloader,这样应该也可以省去原来的zip部分
    """
    sents = [i[0] for i in data]

    token = BertTokenizer.from_pretrained(BERT_PATH)
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
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=64,
                                   return_tensors='pt',
                                   return_length=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    return input_ids, attention_mask, token_type_ids


def predict(test_csv, BERT_PATH, trained_model_path):
    # Step1
    pretrained = BertModel.from_pretrained(BERT_PATH)
    for param in pretrained.parameters():
        param.requires_grad_(False)
    checkpoint = torch.load(trained_model_path)
    EM_model = EM_CLS_Net(pretrained_model=pretrained)
    EM_model.load_state_dict(checkpoint['model_state']) 
    EM_model.eval()

    # Step2
    emotion_test_ds = Emotion_Dataset(test_csv)
    emotion_test_dl = DataLoader( dataset=emotion_test_ds,
                                  batch_size=4,
                                  collate_fn=collate_fn,
                                  shuffle=False,
                                  drop_last=False)
    positivate_out = []
    neutral_out = []
    negative_out = []
    for iter, (input_ids, attention_mask, token_type_ids) in enumerate(emotion_test_dl):
        out = EM_model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        logging.info(f"out is {out}")
        out = torch.squeeze(out).detach().numpy()
        for item in out:
            positivate_out.append(item[0])
            neutral_out.append(item[1])
            negative_out.append(item[2])

    test_csv['positivate'] = positivate_out
    test_csv['neutral'] = neutral_out
    test_csv['negativate'] = negative_out

    return test_csv[['text','positivate','neutral','negativate']]


def predict_one(text, BERT_PATH, trained_model_path):
    # Step1
    pretrained = BertModel.from_pretrained(BERT_PATH)
    for param in pretrained.parameters():
        param.requires_grad_(False)
    checkpoint = torch.load(trained_model_path)
    EM_model = EM_CLS_Net(pretrained_model=pretrained, n_class=2)
    EM_model.load_state_dict(checkpoint['model_state']) 
    EM_model.eval()

    # Step2
    token = BertTokenizer.from_pretrained(BERT_PATH)
    # 单条句子用encode_plus处理
    data = token.encode_plus(text,
                             truncation=True,
                             padding='max_length',
                             max_length=64,
                             )
    # BERT默认有一个维度是batch,必须将encode_plus的结果unsqueeze(0)添加一个维度
    # 参考：https://blog.csdn.net/Ang_Quantum/article/details/121486890
    input_ids = torch.Tensor(data['input_ids']).to(DEVICE, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.Tensor(data['attention_mask']).to(DEVICE, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.Tensor(data['token_type_ids']).to(DEVICE, dtype=torch.long).unsqueeze(0)
    out = EM_model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   token_type_ids=token_type_ids)
    out = torch.squeeze(out).detach().numpy()
    
    return out


def main(args):
    trained_model_path = args.model_path
    if args.predict:
        # 批量预测处理
        test_csv = get_two_classes_csv_data(file_path='data_process/dataset/Automobile_UserPoint_train.csv', file_type='test')
        result = predict(test_csv, BERT_PATH, trained_model_path)
        result.to_csv('result.csv', index=False, encoding='UTF-8')

        return 
    if args.predict_one:
        # 预测单条文本情感(该text可以通过前端文本框获取)
        text = '这车真垃圾'
        out = predict_one(text, BERT_PATH, trained_model_path)
        if out > 0.6:
            print("积极情感")
        else:
            print("消极情感")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--predict_one', action='store_true')
    parser.add_argument('--model_path', type=str, help='Trainned Model Path')
    args = parser.parse_args()
    main(args)
