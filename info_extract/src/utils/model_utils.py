import os
import math
import torch
import torch.nn as nn
from itertools import repeat
from transformers import BertModel
from src.utils.evaluator import span_decode
import pdb

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()


        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target,
                                                                                   reduction=self.reduction,
                                                                                   ignore_index=self.ignore_index)




class BaseModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 dropout_prob):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'
        
        if 'nezha' in bert_dir:
#             pdb.set_trace()
            self.bert_module = NeZhaModel.from_pretrained(bert_dir,
                                                         output_hidden_states=True,
                                                         hidden_dropout_prob=dropout_prob)
        else:
            self.bert_module = BertModel.from_pretrained(bert_dir,
                                                         output_hidden_states=True,
                                                         hidden_dropout_prob=dropout_prob)

        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


# baseline
class CRFModel(BaseModel):
    def __init__(self,
                 bert_dir,
                 num_tags,
                 opt,
                 dropout_prob=0.1,
                 **kwargs):
        super(CRFModel, self).__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)
        self.opt = opt

        out_dims = self.bert_config.hidden_size

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        out_dims = mid_linear_dims

        self.classifier = nn.Linear(out_dims, num_tags)

        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.5)

        self.crf_module = CRF(num_tags=num_tags, batch_first=True)

        init_blocks = [self.mid_linear, self.classifier]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels=None,
                pseudos=None, raw_text=None):

        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 常规
        seq_out = bert_outputs[0]

        seq_out = self.mid_linear(seq_out)

        emissions = self.classifier(seq_out)

        if not self.opt.pseudo:
            pseudos = None

        if labels is not None:
            if pseudos is not None:
                # (batch,)
                tokens_loss = -1. * self.crf_module(emissions=emissions,
                                                    tags=labels.long(),
                                                    mask=attention_mask.byte(),
                                                    reduction='none')

                # nums of pseudo data
                pseudo_nums = pseudos.sum().item()
                total_nums = input_ids.shape[0]

                # learning parameter
                rate = torch.sigmoid(self.loss_weight)
                if pseudo_nums == 0:
                    loss_0 = tokens_loss.mean()
                    loss_1 = (rate * pseudos * tokens_loss).sum()
                else:
                    if total_nums == pseudo_nums:
                        loss_0 = 0
                    else:
                        loss_0 = ((1 - rate) * (1 - pseudos) * tokens_loss).sum() / (total_nums - pseudo_nums)
                    loss_1 = (rate * pseudos * tokens_loss).sum() / pseudo_nums

                tokens_loss = loss_0 + loss_1

            else:
                tokens_loss = -1. * self.crf_module(emissions=emissions,
                                                    tags=labels.long(),
                                                    mask=attention_mask.byte(),
                                                    reduction='mean')

            out = (tokens_loss,)

        else:
            tokens_out = self.crf_module.decode(emissions=emissions, mask=attention_mask.byte())

            out = (tokens_out, emissions)

        return out



class SpanModel(BaseModel):
    def __init__(self,
                 bert_dir,
                 num_tags,
                 opt,
                 dropout_prob=0.1,
                 loss_type='ce',
                 **kwargs):
        """
        tag the subject and object corresponding to the predicate
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        """
        super(SpanModel, self).__init__(bert_dir, dropout_prob=dropout_prob)
        self.opt = opt
        out_dims = self.bert_config.hidden_size

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.num_tags = num_tags

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        out_dims = mid_linear_dims

        self.start_fc = nn.Linear(out_dims, num_tags)
        self.end_fc = nn.Linear(out_dims, num_tags)

        reduction = 'none'
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        else:
            self.criterion = FocalLoss(reduction=reduction)

        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.5)

        init_blocks = [self.mid_linear, self.start_fc, self.end_fc]

        self._init_weights(init_blocks)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                start_ids=None,
                end_ids=None,
                pseudos=None,labels=None):

        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        seq_out = bert_outputs[0]

        seq_out = self.mid_linear(seq_out)

        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)

        out = (start_logits, end_logits, )

        if start_ids is not None and end_ids is not None and self.training:

            start_logits = start_logits.view(-1, self.num_tags)
            end_logits = end_logits.view(-1, self.num_tags)

            # 去掉 padding 部分的标签，计算真实 loss
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]

            pseudos = None
            if pseudos is not None:
                # (batch,)
                start_loss = self.criterion(start_logits, start_ids.view(-1)).view(-1, start_ids.shape[-1]).mean(dim=-1)
                end_loss = self.criterion(end_logits, end_ids.view(-1)).view(-1, start_ids.shape[-1]).mean(dim=-1)

                # nums of pseudo data
                pseudo_nums = pseudos.sum().item()
                total_nums = input_ids.shape[0]

                # learning parameter
                rate = torch.sigmoid(self.loss_weight)
                if pseudo_nums == 0:
                    start_loss = start_loss.mean()
                    end_loss = end_loss.mean()
                else:
                    if total_nums == pseudo_nums:
                        start_loss = (rate * pseudos * start_loss).sum() / pseudo_nums
                        end_loss = (rate * pseudos * end_loss).sum() / pseudo_nums
                    else:
                        start_loss = (rate * pseudos * start_loss).sum() / pseudo_nums \
                                     + ((1 - rate) * (1 - pseudos) * start_loss).sum() / (total_nums - pseudo_nums)
                        end_loss = (rate * pseudos * end_loss).sum() / pseudo_nums \
                                   + ((1 - rate) * (1 - pseudos) * end_loss).sum() / (total_nums - pseudo_nums)
            else:
                start_loss = self.criterion(active_start_logits, active_start_labels).mean()
                end_loss = self.criterion(active_end_logits, active_end_labels).mean()

            loss = start_loss + end_loss

            out = (loss, ) + out

        return out


def build_model(task_type, bert_dir,opt, **kwargs):
    assert task_type in ['crf', 'span', 'mrc']

    if task_type == 'crf':
        model = CRFModel(bert_dir=bert_dir,
                         num_tags=kwargs.pop('num_tags'),opt=opt,
                         dropout_prob=kwargs.pop('dropout_prob', 0.1))

    elif task_type == 'mrc':
        model = MRCModel(bert_dir=bert_dir,
                         dropout_prob=kwargs.pop('dropout_prob', 0.1),
                         opt=opt,
                         use_type_embed=kwargs.pop('use_type_embed'),
                         loss_type=kwargs.pop('loss_type', 'ce'))

    else:
        model = SpanModel(bert_dir=bert_dir,
                          num_tags=kwargs.pop('num_tags'),
                          opt=opt,
                          dropout_prob=kwargs.pop('dropout_prob', 0.1),
                          loss_type=kwargs.pop('loss_type', 'ce'))

    return model
