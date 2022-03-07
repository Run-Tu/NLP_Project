import torch
import torch.nn as nn

class EM_CLS_Net(nn.Module):
    """
        下游:nn.Linear(768,3)
        768：BERT Embedding dim
        3：Emotion classes
    """
    def __init__(self, pretrained_model):
        super(EM_CLS_Net, self).__init__()
        self.pretrained = pretrained_model
        self.fully_connected_layer = nn.Linear(768,3)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out =  self.pretrained(
                            input_ids = input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids
                                  )
        out = self.fully_connected_layer(out.last_hidden_state[:,0])
        out = out.softmax(dim=1)

        return out
