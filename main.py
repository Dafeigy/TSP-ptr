import Models
import torch
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim


# actor = Models.Actor()

beta = 0.9

class TrainPipeline():
    def __init__(self, 
                 model,
                 trainset,
                 valset,
                 batch_size,
                 epoch,
                 optimizer,
                 max_grad_norm,
                 threshold
                 ):
        
        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.batch_size = batch_size
        self.epoch = epoch
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.threshold = threshold
        self.epochs = 0

        self.train_tour = []
        self.val_tour = []

    def train_and_val(self, epoch_nums):
        critic_exp_mvg_avg = torch.zeros(1)
        for epoch in tqdm(range(epoch_nums)):
            # Train Mode
            for batch_id, data in enumerate(self.trainset):
                self.model.train()
                self.optimizer.zero_grad()
                inputs = Variable(data).cuda()

                R, probs, actions, actions_idxs =self.model(inputs)
                if batch_id == 0: 
                    critic_exp_mvg_avg = R.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

                advantange = R - critic_exp_mvg_avg

                logprobs = 0 
                for prob in probs:
                    logprob = torch.log([prob])
                    logprobs += logprob
                logprobs[logprobs < -1000] = 0.

                reinforce = advantange * logprobs
                actor_loss = reinforce.mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                              float(self.max_grad_norm),norm_type=2)
                self.optimizer.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                self.train_tour.append(R.mean().data[0])

                # Eval
                if batch_id % 100 == 0:
                    self.model.eval()
                    for val_data in self.valset:
                        inputs = Variable(val_data).cuda()
                        R, probs, actions, actions_idxs = self.model(inputs)
                        self.val_tour.append(R.mean().data[0])

            if self.threshold and self.train_tour[-1] < self.threshold:
                break

            self.epochs += 1
