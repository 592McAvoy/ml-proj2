import cv2
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import utils.loss as module_loss
import utils.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
import trainer as module_trainer
from utils import prepare_device



# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    tgt_cls = config['target_cls'] if config['target_cls'] > 0 else None
    train_loader = config.init_obj(
        'train_loader', module_data, target_cls=tgt_cls)
    valid_loader = train_loader.split_validation()
    # valid_loader = config.init_obj(
    #     'train_loader', module_data, target_cls=tgt_cls, mode='valid')
    # print(len(train_loader), len(valid_loader))
    # valid_data_loader = data_loader.split_validation()
    # exit()

    # build model architecture, then print to console
    if 'Kernel' in config["name"]:
        model = config.init_obj('arch', module_arch,
                                N=train_loader.get_batchsize())
    else:
        model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # config.init_obj('arch', module_arch)
    trainer = config.init_obj('trainer_module', module_trainer,
                              model, criterion, metrics, optimizer,
                              config=config,
                              device=device,
                              data_loader=train_loader,
                              valid_data_loader=valid_loader,
                              lr_scheduler=lr_scheduler)
    # trainer = Trainer(model, criterion, metrics, optimizer,
    #                   config=config,
    #                   device=device,
    #                   data_loader=train_loader,
    #                   valid_data_loader=valid_loader,
    #                   lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to lavalid checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='train_loader;args;batch_size'),
        CustomArgs(['--num', '--target_cls'], type=int, target='target_cls'),
        CustomArgs(['--spec'], type=str, target='spec'),
        CustomArgs(['--fea', '--fea_base'], type=int, target='arch;args;fea_base'),
        CustomArgs(['--layer', '--n_layer'], type=int, target='arch;args;n_layer')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
