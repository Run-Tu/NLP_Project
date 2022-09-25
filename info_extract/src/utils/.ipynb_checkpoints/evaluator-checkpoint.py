import torch
import logging
import numpy as np
from collections import defaultdict
import pdb

logger = logging.getLogger(__name__)


def get_base_out(model, loader, device):
    """
    每一个任务的 forward 都一样，封装起来
    """
    model.eval()

    with torch.no_grad():
        for idx, _batch in enumerate(loader):

            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)

            tmp_out = model(**_batch)

            yield tmp_out

def model_evaluate(model,dev_load,opt,device,ent2id):
    f = open(file='./tmp_dev_evaluate_{}'.format(opt.task_type),mode='w',encoding='utf-8')
    id2ent = {v:k for k,v in ent2id.items()}
    model.eval()
    with torch.no_grad():
        for batch,batch_data in enumerate(dev_load):
            raw_text = batch_data['raw_text']
            del batch_data['raw_text']
            labels = batch_data['bieo_labels']
            del batch_data['bieo_labels']

            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)

            if opt.task_type == 'crf':
                decode_output = model(**batch_data)[0]
                labels = [list(sample[1:-1]) for sample in labels.numpy()]
                decode_output = [sample[1:-1] for sample in decode_output]
                for text,decode,label in zip(raw_text,decode_output,labels):
                    tmp_decode = [id2ent[x] for x in decode]
                    tmp_label = [id2ent[x] for x in label][:len(tmp_decode)]
                    assert len(text)==len(tmp_decode)
                    for char,true,pre in zip(text,tmp_label,tmp_decode):
                        f.write('{}\n'.format(' '.join([char,true,pre])))
                    f.write('\n')

            if opt.task_type == 'span':
                decode_output = model(**batch_data)
                # labels = [sample[1:-1] for sample in labels]
                start_logits = decode_output[0].cpu().numpy()
                end_logits = decode_output[1].cpu().numpy()
                for tmp_start_logits, tmp_end_logits,text,label in zip(start_logits,end_logits,raw_text,labels):
                    tmp_start_logits = tmp_start_logits[1:1 + len(text)]
                    tmp_end_logits = tmp_end_logits[1:1 + len(text)]
                    predict = span_decode(tmp_start_logits,tmp_end_logits,text,id2ent)
                    tmp_label = label[:len(text)]
                    for char,true,pre in zip(text,tmp_label,predict):
                        f.write('{}\n'.format(' '.join([char,true,pre])))
                    f.write('\n')
    f.close()




def span_decode(start_logits, end_logits, raw_text, id2ent):
    predict=[]
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

    result = ['O']*len(raw_text)
    for item in tmp:
        s, e, flag = item[1], item[2], id2ent[item[3]]
        if e>s:
            result[s]='B-{}'.format(flag)
            result[e]='E-{}'.format(flag)
            if e-s>1:
                for i in range(s+1,e):
                    result[i]='I-{}'.format(flag)
        if e==s:
            result[s] = 'S-{}'.format(flag)


    return result