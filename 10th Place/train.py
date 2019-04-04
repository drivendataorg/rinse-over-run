import argparse
import gc
import os

import jstyleson

from data_loader import RinseDataLoader
from evaluation.inference import do_inference
from logger import Logger
from model.model import *
from trainer import TrainerRNNLSTM
from model.loss import *


def main(config, resume):
    train_logger = Logger()

    loss = eval(config['loss'])

    train_data_folder = config['data_loader']['data_dir_train']
    test_data_folder = config['data_loader']['data_dir_control']

    models = []

    train_logger.logger.info('Create test loader')

    for fold in range(config['n_splits']):
        model = new_model(config)

        models.append(model)

    if config['only_inference']:
        inference_only_weights_folder = 'saved/submission_best'
        files_with_weights = []
        for i in range(config['n_splits']):
            files_with_weights.append(os.path.join(inference_only_weights_folder, '%d-model_best.pth.tar' % i))
    else:
        files_with_weights = train_n_splits(config, loss, resume, train_data_folder, train_logger)

    do_inference(config, models=models, results_file_name='test', inbound_data_folder=test_data_folder,
                 files_with_weights=files_with_weights)


def train_n_splits(config, loss, resume, train_data_folder, train_logger):
    files_with_weights = []
    for fold in range(config['n_splits']):
        print('fold = ', fold)
        gc.collect()
        train_for_fold(config, fold, loss, resume, train_data_folder, train_logger)
        files_with_weights.append('saved/%s/%d-model_best.pth.tar' % (config['name'], fold))

    return files_with_weights


def train_for_fold(config, fold, loss, resume, train_data_folder, train_logger):
    model = new_model(config)
    if not config['only_inference']:
        train_logger.logger.info('Create train loader')
        train_data_loader = RinseDataLoader(config, name='train', inbound_folder=train_data_folder,
                                            fold_number=fold, evaluate=False)
        train_a_model(config, loss, model, resume, train_data_loader, train_logger)
        print('start validation')
        train_data_loader = None
        gc.collect()
        do_inference(config, [model], results_file_name='train', inbound_data_folder=train_data_folder,
                     files_with_weights=['saved/%s/%d-model_best.pth.tar' % (
                         config['name'],
                         fold)], fold_number=fold)


def train_a_model(config, loss, model, resume, train_data_loader, train_logger):
    train_logger.logger.info('Create trainer')
    trainer = TrainerRNNLSTM(model, loss,
                             resume=resume,
                             config=config,
                             data_loader=train_data_loader,
                             train_logger=train_logger)
    train_logger.logger.info('Start training')
    trainer.train()


def new_model(config):
    model = RinseModel(config=config, model_name='LSTM_and_4_linear')
    return model


if __name__ == '__main__':
    logger = Logger(__name__).logger

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = jstyleson.load(open(args.config))
    assert config is not None

    main(config, args.resume)
