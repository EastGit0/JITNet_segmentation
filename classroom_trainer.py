import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import ClassroomBaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm

class ClassroomTrainer(ClassroomBaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(ClassroomTrainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)

        self.wrt_mode, self.wrt_step = 'train_', 0
        # self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        # if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        # self.num_classes = self.train_loader.dataset.num_classes

        # if self.device ==  torch.device('cpu'): prefetch = False
        # if prefetch:
        #     self.train_loader = DataPrefetcher(train_loader, device=self.device)
        #     # self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch, train_loader):
        if epoch == 1:
            print("Prefetching dataloader")
            self.train_loader = train_loader
            self.num_classes = train_loader.dataset.num_classes

            if self.device ==  torch.device('cpu'): prefetch = False
            if prefetch:
                self.train_loader = DataPrefetcher(train_loader, device=self.device)
                # self.val_loader = DataPrefetcher(val_loader, device=self.device)

        print("Starting Epoch!")
        self.logger.info('\n')

        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            #data, target = data.to(self.device), target.to(self.device)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            loss, seg_metrics = self.model(data, target)
 
            if isinstance(self.model, torch.nn.DataParallel):
                #num_gpus = loss.shape[0]
                num_gpus = 1
                loss = loss.mean()

            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # PRINT INFO
            # tbar.set_description('TRAIN ({}) | Loss: {:.3f} | B {:.2f} D {:.2f} |'.format(
            #                                     epoch, self.total_loss.average,
            #                                     self.batch_time.average, self.data_time.average))
            print(tbar.set_description('TRAIN ({}) | Loss: {:.3f} | B {:.2f} D {:.2f} |'.format(
                                                epoch, self.total_loss.average,
                                                self.batch_time.average, self.data_time.average)))

        # RETURN LOSS & METRICS
        # log = {'loss': self.total_loss.average,
        #         **seg_metrics}

        #if self.lr_scheduler is not None: self.lr_scheduler.step()
        return 0

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
