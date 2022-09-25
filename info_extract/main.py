import time
import os
import json
import logging
from torch.utils.data import DataLoader
from src.utils.trainer import train
from src.utils.options import Args
from src.utils.model_utils import build_model
from src.utils.dataset_utils import NERDataset
from src.utils.functions_utils import set_seed,Data_Processor
import sys
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def train_base(opt, train_feature,dev_feature=None,test_feature=None):
    #加载实体映射表
    with open(os.path.join(opt.raw_data_dir, f'{opt.task_type}_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)


    train_dataset = NERDataset(train_feature,opt,ent2id)
    dev_dataset = NERDataset(dev_feature,opt,ent2id)


    if opt.task_type == 'crf':
        model = build_model('crf', opt.bert_dir,opt, num_tags=len(ent2id),
                            dropout_prob=opt.dropout_prob)
    elif opt.task_type == 'mrc':
        model = build_model('mrc', opt.bert_dir,opt,
                            dropout_prob=opt.dropout_prob,
                            use_type_embed=opt.use_type_embed,
                            loss_type=opt.loss_type)
    else:
        model = build_model('span', opt.bert_dir,opt, num_tags=len(ent2id)+1,
                            dropout_prob=opt.dropout_prob,
                            loss_type=opt.loss_type)
    
    logger.info("make train......")
    train(opt, model, train_dataset,dev_dataset,ent2id)

    

def training(opt):
    processor = Data_Processor(opt.raw_data_dir)
    train,dev = processor.get_label_data()
    train_base(opt,train,dev)


if __name__ == '__main__':

    args = Args().get_parser()
    set_seed(args.seed)
    training(args)
