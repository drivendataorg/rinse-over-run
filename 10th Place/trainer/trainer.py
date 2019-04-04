import datetime
import os

import numpy as np
import torch

from base import BaseTrainer, print_training_log


class TrainerRNNLSTM(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, resume, config,
                 data_loader, train_logger=None):
        super(TrainerRNNLSTM, self).__init__(model, loss, metrics=None,
                                             resume=resume, config=config, train_logger=train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.log_step = int(np.sqrt(self.batch_size))
        self.device = self.config['device']
        self.fold = self.data_loader.dataset.fold_number

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        """
        start = datetime.datetime.now()
        self.model.train()
        total_loss = 0
        for batch_idx, data in (enumerate(self.data_loader)):
            # print('data ', data)
            features, lengths, metas, targets, names = data
            # features, targets, names = data

            features = features.to(device=torch.device(self.device))
            metas = metas.to(device=torch.device(self.device))
            targets = targets.to(device=torch.device(self.device))

            targets_pred = self.model((features, lengths, metas))
            loss = self.loss(targets_pred, targets, self.config)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            print_training_log(batch_idx, epoch, loss, self.verbosity, self.logger, self.log_step, self.data_loader)

        log = {
            'loss': total_loss / len(self.data_loader)
        }
        end = datetime.datetime.now()
        print(epoch, ' Time ', end - start, ' loss ', total_loss / len(self.data_loader))
        return log

    def _save_checkpoint(self, epoch, log, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, '{:1d}-checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'
                                .format(self.fold, epoch, log['loss']))
        torch.save(state, filename)
        if save_best:
            os.rename(filename, os.path.join(self.checkpoint_dir, '{:1d}-model_best.pth.tar'.format(self.fold)))
            self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))
        else:
            self.logger.info("Saving checkpoint: {} ...".format(filename))
