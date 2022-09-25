import torch
import torch.nn as nn

class EM_CLS_Net(nn.Module):
    """
        下游:nn.Linear(768,3)
        768：BERT Embedding dim
        3：Emotion classes
    """
    def __init__(self, pretrained_model, n_class):
        super(EM_CLS_Net, self).__init__()
        self.pretrained = pretrained_model
        self.n_class = n_class
        if self.n_class == 2:
            self.fully_connected_layer = nn.Linear(768,1)
        else:
            self.fully_connected_layer = nn.Linear(768,n_class)
    
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out =  self.pretrained(
                            input_ids = input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids
                                  )
        out = self.fully_connected_layer(out.last_hidden_state[:,0])
        if self.n_class == 2:
            out = torch.sigmoid(out)
        else:
            out = out.softmax(dim=1)

        return out
