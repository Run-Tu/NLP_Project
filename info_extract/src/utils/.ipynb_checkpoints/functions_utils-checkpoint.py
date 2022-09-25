import os
import copy
import torch
import random
import numpy as np
from collections import defaultdict,Counter
from datetime import timedelta
from src.utils.dataset_utils import NERDataset
import time
from torch.utils.data import DataLoader, RandomSampler
from src.utils.model_utils import build_model
import logging
import pdb
import json
import jieba
from tqdm import tqdm
logger = logging.getLogger(__name__)

def set_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """

    # set to device to the first cuda
    device = torch.device("cpu" if gpu_ids == '-1' else "cuda:" + gpu_ids[0])

    if ckpt_path is not None:
        logger.info(f'Load ckpt from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    model.to(device)

    if gpu_ids!='-1' and len(gpu_ids) > 1:
        logger.info(f'Use multi gpus in: {gpu_ids}')
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info(f'Use single gpu in: {gpu_ids}')

    return model, device


def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    tmp = os.listdir(base_dir)
    tmp = ['checkpoint_{}_epoch.pt'.format(i) for i in range(21) if 'checkpoint_{}_epoch.pt'.format(i) in tmp]
    model_lists = [os.path.join(base_dir,x) for x in tmp]
    return model_lists


def swa(model, model_dir):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)
    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list:
            logger.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1
    return swa_model


class Data_Processor():
    def __init__(self,path):
        self.path = path

    def get_label_data(self,):
        data = json.load(open(os.path.join(self.path,"train.json"),encoding="utf-8"))
        res = []
        for item in data:
            text,stock_names = item["text"],item["stock_name"]
            tmp_text = text
            tmp_label = []
            for stock_name in stock_names:
                start_pos = tmp_text.find(stock_name)
                end_pos = start_pos + len(stock_name) - 1
                tmp_label.append((stock_name,start_pos,end_pos))
                tmp_text = tmp_text.replace(stock_name, "$" * len(stock_name), 1)
            res.append({"text":text,"stock_name":tmp_label})

        train = [x for i,x in enumerate(res) if i%10!=0]
        dev = [x for i,x in enumerate(res) if i%10==0]
        return train,dev


def infer_span_decode(start_logits, end_logits, raw_text, id2ent):
    predict=[]
    entity = []
    start_pred = np.argmax(start_logits, -1)
    end_pred = np.argmax(end_logits, -1)

    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i:i + j + 1]
                predict.append((''.join(tmp_ent),i,i+j,s_type))
                break
    tmp = []
    for item in predict:
        if not tmp:
            tmp.append(item)
        else:
            if item[1]>tmp[-1][2]:
                tmp.append(item)

    predict = [x[0] for x in tmp]
    return predict


    
    

