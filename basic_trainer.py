import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from utils import static_utils
import logging
import os
import scipy

from SAM_function.DREAM import DREAM
from SAM_function.FSAM import FSAM

class BasicTrainer:
    def __init__(self, model, epoch_threshold = 150, model_name='NeuroMax', use_SAM=1, SAM_name='DREAM', epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5, 
                    rho = 0.005, device='cuda', sigma=0.1, lmbda=0.9, acc_step=8):
        self.model = model
        self.epoch_threshold = epoch_threshold
        self.model_name = model_name
        self.SAM_name = SAM_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.use_SAM = use_SAM

        self.rho = rho 
        self.device = device
        self.sigma = sigma
        self.lmbda = lmbda
        self.acc_step = acc_step
        self.logger = logging.getLogger('main')

    def make_adam_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer
    

    def make_sam_optimizer(self,):
        base_optimizer = torch.optim.SGD
        if self.SAM_name == 'FSAM':
            optimizer = FSAM(
                self.model.parameters(),
                base_optimizer, device=self.device,
                lr=self.learning_rate,
                sigma=self.sigma, lmbda=self.lmbda
                )
        elif self.SAM_name == 'DREAM':
            optimizer = DREAM(
                self.model.parameters(),
                base_optimizer, device=self.device,
                lr=self.learning_rate,
                sigma=self.sigma, lmbda=self.lmbda
                )
        else:
            print("WRONG!!")
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
    
        if self.model_name == 'FASTopic':
            train_theta = self.test(dataset_handler.train_contextual_embed, dataset_handler.train_contextual_embed)
        else:
            train_theta = self.test(dataset_handler.train_data)

        return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        accumulation_steps = self.acc_step
        adam_optimizer = self.make_adam_optimizer()
        sam_optimizer = self.make_sam_optimizer() 

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(adam_optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)
        if self.use_SAM == 0:
            print("Donot use SAM")

        for epoch_id, epoch in enumerate(tqdm(range(1, self.epochs + 1))):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_id, batch in enumerate(dataset_handler.train_dataloader): 
                *inputs, indices = batch
                batch_data = inputs
                # rst_dict = self.model(indices, is_OT, batch_data, epoch_id=epoch)
                rst_dict = self.model(indices, batch_data, epoch_id=epoch)
                batch_loss = rst_dict['loss']
                batch_loss.backward()

                # batch_data_tensor = torch.tensor(batch_data, dtype=torch.float32)
                # theta = self.model.get_theta(batch_data_tensor)

                if self.use_SAM == 0:
                    adam_optimizer.step()
                    adam_optimizer.zero_grad()
                else:
                    #if (batch_id + 1) % accumulation_steps == 0 or (batch_id + 1) == len(dataset_handler.train_dataloader):
                    if epoch_id > self.epoch_threshold:
                        #theta, _ = self.model.encode(batch_data[0].to('cuda'))
                        #loss_OT_ = self.model.get_loss_OT(theta, indices)
                        
                        if self.SAM_name == 'DREAM':
                            self.model.is_OT = False
                            loss_OT_ = self.model.get_loss_OT(batch_data, indices)
                            sam_optimizer.first_step(loss_OT_,
                                                    zero_grad=True)
                        else:
                            sam_optimizer.first_step(zero_grad=True)

                        # rst_dict_adv = self.model(indices, is_OT, batch_data, epoch_id=epoch)
                        rst_dict_adv = self.model(indices, batch_data, epoch_id=epoch)

                        batch_loss_adv = rst_dict_adv['loss']
                        batch_loss_adv.backward()

                        sam_optimizer.second_step(zero_grad=True)
                    
                    else:
                        if self.SAM_name == 'DREAM':
                            self.model.is_OT = True
                        adam_optimizer.step()
                        adam_optimizer.zero_grad()
                    

                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * \
                            len(batch_data['data'])
                    except:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data)


            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                #print(output_log)
                self.logger.info(output_log)

    def test(self, input_data, train_data=None):
        data_size = input_data.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
            
                if self.model_name == 'FASTopic':
                    batch_theta = self.model.get_theta(batch_input, train_data)
                else:
                    batch_theta = self.model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        if self.model_name == 'FASTopic':
            train_theta = self.test(dataset_handler.train_contextual_embed, dataset_handler.train_contextual_embed)
            test_theta = self.test(dataset_handler.test_contextual_embed, dataset_handler.train_contextual_embed)
        else:
            train_theta = self.test(dataset_handler.train_data)
            test_theta = self.test(dataset_handler.test_data)
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        if hasattr(self.model, 'group_embeddings'):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
            self.logger.info(
                f'group_embeddings size: {group_embeddings.shape}')

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

        return word_embeddings, topic_embeddings
