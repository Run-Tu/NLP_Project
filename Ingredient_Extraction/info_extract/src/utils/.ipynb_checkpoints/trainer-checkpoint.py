import os
import copy
import torch
import logging
from torch.cuda.amp import autocast as ac
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from src.utils.attack_train_utils import FGM, PGD
from src.utils.functions_utils import load_model_and_parallel, swa
from src.utils.evaluator import model_evaluate
from src.utils.conlleval import calculte_metrics
from src.utils.dataset_utils import infer
import pdb
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_model(opt, model,epoch):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    torch.save(model_to_save.state_dict(), os.path.join(opt.output_dir,'best_model.pt'))


def build_optimizer_and_scheduler(opt, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))


    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def train(opt, model, train_dataset,dev_dataset,ent2id):
    if opt.task_type in ['span','crf']:
        fn = train_dataset.collate_fn
    else:
        fn = train_dataset.collate_fn_mrc
        
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt.train_batch_size,
                              num_workers=0,
                              collate_fn=fn,
                              shuffle=True)

    dev_loader = DataLoader(dataset=dev_dataset,
                              batch_size=opt.train_batch_size,
                              num_workers=0,
                              collate_fn=fn,
                              shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    swa_raw_model = copy.deepcopy(model)


    t_total = len(train_loader) * opt.train_epochs

    optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)


    logger.info("batch_size:{},epochs:{},train_nums:{},dev_nums:{}".format(opt.train_batch_size,
                                                                           opt.train_epochs,
                                                                           len(train_dataset),
                                                                           len(dev_dataset)))


    model.zero_grad()

    fgm, pgd = None, None

    attack_train_mode = opt.attack_train.lower()
    if attack_train_mode == 'fgm':
        fgm = FGM(model=model)
    elif attack_train_mode == 'pgd':
        pgd = PGD(model=model)

    pgd_k = 3
    avg_loss = 0.
    best_f1 = 0
    best_model = None

    for epoch in range(opt.train_epochs):
        torch.cuda.empty_cache()

        for step, batch_data in enumerate(tqdm(train_loader)):
            model.train()
            del batch_data['raw_text']
            del batch_data['bieo_labels']
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            loss = model(**batch_data)[0]
            loss.backward()
            if fgm is not None:
                fgm.attack()
                loss_adv = model(**batch_data)[0]

                loss_adv.backward()
                fgm.restore()

            elif pgd is not None:
                pgd.backup_grad()

                for _t in range(pgd_k):
                    pgd.attack(is_first_attack=(_t == 0))

                    if _t != pgd_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()

                    loss_adv = model(**batch_data)[0]

                    loss_adv.backward()

                pgd.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            optimizer.step()

            scheduler.step()
            model.zero_grad()

            avg_loss += loss.item()
        
        if opt.task_type=="mrc":
            model_evaluate_mrc(model, dev_loader, opt, device, ent2id)
        else:
            f1 = model_evaluate(model, dev_loader, opt, device, ent2id)
                
        f1 = calculte_metrics(opt)

        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model)
            save_model(opt, best_model, epoch)
    # clear cuda cache to avoid OOM
    torch.cuda.empty_cache()
    logger.info('Train done')