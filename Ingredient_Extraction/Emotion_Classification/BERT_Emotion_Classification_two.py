import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_process.data_process import get_two_classes_csv_data
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
                                   max_length=128,
                                   return_tensors='pt',
                                   return_length=True)

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.Tensor(labels)

    return input_ids, attention_mask, token_type_ids, labels  


def main():
    train_data = get_two_classes_csv_data(file_path='data_process/dataset/2_classes/weibo_senti_100k.csv', file_type='train')
    valid_data = get_two_classes_csv_data(file_path='data_process/dataset/2_classes/weibo_senti_100k.csv', file_type='valid')
    emotion_train_ds = Emotion_Dataset(train_data, n_class=2)
    emotion_valid_ds = Emotion_Dataset(valid_data, n_class=2)
    emotion_train_dl = DataLoader(dataset=emotion_train_ds,
                                  batch_size=16,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=False)
    emotion_valid_dl = DataLoader(dataset=emotion_valid_ds,
                                  batch_size=16,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=False)
    pretrained = BertModel.from_pretrained(BERT_PATH)
    for param in pretrained.parameters():
        param.requires_grad_(False)
    EM_model = EM_CLS_Net(pretrained_model=pretrained, n_class=2)
    optimizer = AdamW(EM_model.parameters(), lr=5e-4, weight_decay=0.01)

    trainner =  Trainner()
    trainner.training(
                        model = EM_model,
                        device = DEVICE,
                        epochs = 4,
                        train_dl = emotion_train_dl,
                        valid_dl = emotion_valid_dl,
                        n_class = 2,
                        optimizer = optimizer
                    )


if __name__ == '__main__':
        main()