import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
from src.utils.evaluator import span_decode
import numpy as np
import random



class NERDataset(Dataset):
    def __init__(self,train_feature,opt,ent2id):
        self.data = train_feature
        self.nums = len(train_feature)
        self.tokenizer = BertTokenizer(os.path.join(opt.bert_dir, 'vocab.txt'))
        self.ent2id = ent2id
        self.opt = opt


    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        return self.data[index]
    
    def get_bieo_data(self,text,data):
        labels = ['O']*len(text)
        for _,s,e in data:
            labels[s]="B-stock_name"
            labels[e] = "E-stock_name"
            for i in range(s+1,e):
                labels[i] = "I-stock_name"
        return labels

    def collate_fn(self,batch_data):
        max_len = max([len(x["text"]) for x in batch_data])+2
        input_ids,token_type_ids,attention_mask,labels,raw_text = [],[],[],[],[]
        start_ids,end_ids,bieo_labels = [],[],[]
        for sample in batch_data:
            text = sample["text"]
            label = sample["stock_name"]
            encode_dict = self.tokenizer.encode_plus(text=list(text),
                                                max_length=max_len,
                                                pad_to_max_length=True,
                                                is_pretokenized=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True)
            input_ids.append(encode_dict['input_ids'])
            token_type_ids.append(encode_dict['token_type_ids'])
            attention_mask.append(encode_dict['attention_mask'])
            raw_text.append(text)


            if label and self.opt.task_type == 'crf':
                tmp_label = [self.ent2id[x] for x in label]
                tmp_label = [0]+tmp_label+[0]
                if len(tmp_label)<max_len:
                    padding_len = max_len -len(tmp_label)
                    tmp_label = tmp_label+[0]*padding_len
                labels.append(tmp_label)

            if self.opt.task_type == 'span':
                start_id,end_id = [0]*len(text),[0]*len(text)
                bieo_label = self.get_bieo_data(text,label)
                bieo_labels.append(bieo_label)
                for _,s,e in label:
                    end_id[e] = self.ent2id["stock_name"]
                    start_id[s] = self.ent2id["stock_name"]
                #add CLS、SEP
                start_id = [0]+start_id+[0]
                end_id = [0]+end_id+[0]
                #padding
                if len(start_id)<max_len:
                    start_id = start_id + [0]*(max_len-len(start_id))
                    end_id = end_id + [0]*(max_len-len(end_id))

                start_ids.append(start_id)
                end_ids.append(end_id)

        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).float()
        start_ids = torch.tensor(start_ids).long()
        end_ids = torch.tensor(end_ids).long()


        if self.opt.task_type == 'crf':
            result = ['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'pseudos', 'raw_text']
            return dict(zip(result,[input_ids,token_type_ids,attention_mask,labels,pseudos,raw_text]))
        if self.opt.task_type == 'span':
            result = ['input_ids', 'token_type_ids','attention_mask', 'raw_text','start_ids','end_ids','bieo_labels']
            return dict(zip(result,[input_ids, token_type_ids, attention_mask, raw_text,start_ids,end_ids,bieo_labels]))


def infer(model,dev_load,opt,device,ent2id):
    if opt.cv_num==-1:
        f = open(file='./result.txt', mode='w', encoding='utf-8')
    else:
        f = open(file='./cv_tmp/temp_result_{}'.format(opt.cv_num), mode='w', encoding='utf-8')
    with open(file='./tcdata/final_test.txt',mode='r',encoding='utf-8') as files:
        fu_raw_texts = []
        for line in files:
            fu_raw_texts.append(line.strip())

    with open(file='./data/raw_data/final_test.txt',mode='r',encoding='utf-8') as files:
        chu_raw_texts = []
        for line in files:
            chu_raw_texts.append(line.strip())

    if opt.cv_num==-1:
        raw_texts = fu_raw_texts
    else:
        if opt.cv_infer:
            raw_texts = fu_raw_texts
        else:
            raw_texts = fu_raw_texts + chu_raw_texts

    id2ent = {v:k for k,v in ent2id.items()}
    model.eval()
    decode_output = []
    with torch.no_grad():
        for batch,batch_data in enumerate(dev_load):
            raw_text = batch_data['raw_text']
            del batch_data['raw_text']
            labels = batch_data['labels']
            del batch_data['labels']

            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)

            if opt.task_type == 'crf':
                tmp_decode = model(**batch_data)[0]
                tmp_decode = [sample[1:-1] for sample in tmp_decode]
                decode_output+=tmp_decode

            if opt.task_type == 'span':
                tmp_decode = model(**batch_data)
                start_logits = tmp_decode[0].cpu().numpy()
                end_logits = tmp_decode[1].cpu().numpy()
                for tmp_start_logits, tmp_end_logits,text in zip(start_logits,end_logits,raw_text):
                    tmp_start_logits = tmp_start_logits[1:1 + len(text)]
                    tmp_end_logits = tmp_end_logits[1:1 + len(text)]
                    predict = span_decode(tmp_start_logits,tmp_end_logits,text,id2ent)
                    decode_output.append(predict)


    for text, decode in zip(raw_texts, decode_output):
        tmp_decode_output = " ".join([id2ent[x] if opt.task_type=='crf' else x  for x in decode])
        f.write('{}\n'.format('\u0001'.join([text, tmp_decode_output])))
    f.close()