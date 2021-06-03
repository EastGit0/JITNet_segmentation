import os
import logging
import json
import math
import torch
import datetime
from torch.utils import tensorboard
from utils import helpers
from utils import logger
# import utils.lr_scheduler
from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class ClassroomBaseTrainer:
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None):
        self.model = model
        self.loss = loss
        self.config = config
        # self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1
        self.improved = False
        self.train_count = 0

        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        self.model.loss = loss
        if config["use_synch_bn"]:
            self.model = convert_model(self.model)
            self.model = DataParallelWithCallback(self.model, device_ids=availble_gpus)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
        
        if availble_gpus:
            self.model.cuda()

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        # OPTIMIZER
        if self.config['optimizer']['differential_lr']:
            if isinstance(self.model, torch.nn.DataParallel):
                trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.module.get_decoder_params())},
                                    {'params': filter(lambda p:p.requires_grad, self.model.module.get_backbone_params()),
                                    'lr': config['optimizer']['args']['lr'] / 10}]
            else:
                trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.get_decoder_params())},
                                    {'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()),
                                    'lr': config['optimizer']['args']['lr'] / 10}]
        else:
            trainable_params = filter(lambda p:p.requires_grad, self.model.parameters())
        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        # self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler']['type'])(self.optimizer, self.epochs, len(train_loader))
        #self.lr_scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])(self.optimizer, **config['lr_scheduler']['args'])

        # MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = cfg_trainer.get('early_stop', math.inf)

        # CHECKPOINTS & TENSOBOARD
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], self.config['name'])
        helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        if resume: self._resume_checkpoint(resume)

        if not availble_gpus:
            torch.set_flush_denormal(True)
            print("Flush Denormals Enabled")

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu

        device = torch.device('cuda' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def train(self, train_loader):
        # self.train_loader.dataset._set_files()
        self.train_count = self.train_count + 1
        for epoch in range(self.start_epoch, self.epochs+1):
            # RUN TRAIN (AND VAL)
            results = self._train_epoch(epoch, train_loader)
            # self.lr_scheduler.step()

        self._save_checkpoint(epoch, save_best=False)
        
        return self.train_count

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'state_dict': self.model.state_dict(),
        }
        filename = os.path.join(self.checkpoint_dir, "weights_{}.pth".format(str(self.train_count)))
        self.logger.info(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path, map_location=self.device)

        self.start_epoch = 1
        self.mnt_best = 0
        self.not_improved_count = 0

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError


