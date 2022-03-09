import os
import time
import torch
import numpy as np
from utils.logging_util import get_logging
from torch.utils.tensorboard import SummaryWriter

# get_tensorboard_SummaryWriter
TRAIN_TIME = time.strftime("%Y-%m-%d", time.localtime())
TENSORBOARD_PATH = f'output/loss/tensorboard/{TRAIN_TIME}/'
if not os.path.exists(TENSORBOARD_PATH):
    os.makedirs(TENSORBOARD_PATH)
writer = SummaryWriter(TENSORBOARD_PATH)

# logging
LOG_DIR = 'output/train_log/'
logging = get_logging(LOG_DIR)


class Trainner():
    """
        Trainner()封装训练器功能
    """
    def __init__(self):

        pass


    def save_checkpoint(self, epoch, min_val_loss, model_state, opt_state):
        """
            pytorch实现断点训练参考：https://zhuanlan.zhihu.com/p/133250753
        """
        print(f"New minimum reached at epoch #{epoch+1}, saving model state...")
        checkpoint = {
            'epoch' : epoch+1,
            'min_val_loss' : min_val_loss,
            'model_state' : model_state,
            'opt_state' : opt_state
        }
        
        torch.save(checkpoint, f'output/two_classes_checkpoints/{TRAIN_TIME}_model_state.pt')
    

    def load_checkpoint(self, path, model, optimizer):
        # load check point 
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_val_loss']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])

        return model, optimizer, epoch, min_val_loss

    
    def training(self, model, device, epochs, train_dl, valid_dl, optimizer, n_class, validate_every=1):
        """
            training所需输入：
            model --> 下游任务model
            device --> 设备cpu or gpu
            epochs --> training epochs
            train_dl --> training dataloader
            valid_dl --> validation dataloader
            optimizer --> 优化器
            validate_every --> 每几个epoch进行一次validation

            three steps：
            1. optimizer.zero_grad() 先将梯度归零
            2. loss.backward() 计算每个参数的梯度值
            3. optimizer.step() 通过梯度下降调整参数实现参数更新
        """
        if n_class == 2:
            criterion = torch.nn.BCELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        # save model according to min_validation_loss
        min_validation_loss = np.inf
        # 模型中有BN层(Batch Normalization)和Dropout,需要在训练时添加model.train()
        model.train()
        logging.info("Start")
        for epoch in range(epochs):
            logging.info("**********Model Training**********")
            running_training_loss = 0.0
            # Training
            for iter, (input_ids_batch, attention_mask_batch, token_type_ids_batch, labels_batch) in enumerate(train_dl):
                    #Step1
                    optimizer.zero_grad()
                    # Calculate output
                    out = model(input_ids = input_ids_batch,
                                attention_mask = attention_mask_batch,
                                token_type_ids = token_type_ids_batch
                                )
                    logging.info(f"out is {out}")
                    logging.info(f"labels_batch is {labels_batch}")
                    # Calculate Training loss
                    if n_class == 2:
                        training_loss = criterion(torch.squeeze(out), labels_batch)
                    else:
                        training_loss = criterion(out, labels_batch)
                    logging.info(f"batch_loss is {training_loss}")
                    # Step2&Step3
                    training_loss.backward()
                    optimizer.step()

                    running_training_loss += training_loss.item()
                                                                                                        
            writer.add_scalar('training loss', running_training_loss / len(train_dl), epoch)

            if epoch % validate_every == 0:
                logging.info("**********Model Validation**********")
                # Set to eval mode
                model.eval()
                running_validation_loss = 0.0
                
                for iter, (input_ids_batch, attention_mask_batch, token_type_ids_batch, labels_batch) in enumerate(valid_dl):
                        #Step1
                        optimizer.zero_grad()
                        # Calculate output
                        out = model(input_ids = input_ids_batch,
                                    attention_mask = attention_mask_batch,
                                    token_type_ids = token_type_ids_batch
                                    )
                        # Calculate validation loss
                        if n_class == 2:
                            validation_loss = criterion(torch.squeeze(out), labels_batch)
                        else:
                            validation_loss = criterion(out, labels_batch)
                        running_validation_loss += validation_loss.item()
            # Visualization
            writer.add_scalar('validation loss', running_validation_loss / len(valid_dl), epoch)

            is_best = running_validation_loss / len(list(valid_dl)) <= min_validation_loss

            if is_best:
                min_validation_loss = running_validation_loss / len(valid_dl)
                self.save_checkpoint(
                                     epoch+1, 
                                     min_validation_loss, 
                                     model.state_dict(),
                                     optimizer.state_dict() 
                                    )  
        logging.info("End")
