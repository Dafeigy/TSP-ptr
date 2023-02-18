import Models


actor = Models.Actor()

class TrainModel():
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
        self.threshld = threshold

    def train_and_val(self, epoch_nums):
        for epoch in range(epoch_nums):
            # Train Mode
            for batch_id, data in enumerate(self.trainset):
                self.model.train()

            # Eval mode
            for bat_id, data in enumerate(self.valset):
                self.model.eval()
