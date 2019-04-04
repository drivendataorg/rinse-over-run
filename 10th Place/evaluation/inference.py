import datetime
import os

import torch
# https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit/38739634#38739634
from tqdm import tqdm

from data_loader import RinseDataLoader
from evaluation.write_results import write_results_in_my_format, remove_directory
from logger import Logger
from model.loss import *
from model.model import CompoundModel


def outputs_for_large_dataset(loader, network):
    config = loader.config
    name = loader.dataset.name
    batches_number = save_inference_results_on_disk(loader, network, name)
    return read_inference_results_from_disk(config, batches_number, name)


def read_inference_results_from_disk(config, batches_number, name):
    path = os.path.join(config['temp_folder'], name, '')
    pack_volume = config['pack_volume']
    assert 'all_outputs_%d' % pack_volume in os.listdir(path), \
        'There should be precomputed inference data in %s!' % path

    all_outputs = []
    all_names = []
    all_targets = []
    for i in range(1, batches_number + 1):
        outputs = torch.load('%sall_outputs_%d' % (path, i * pack_volume))
        names = torch.load('%sall_names_%d' % (path, i * pack_volume))
        targets = torch.load('%sall_targets_%d' % (path, i * pack_volume))
        all_outputs.extend(outputs)
        all_names.extend(names)
        all_targets.extend(targets)

    return all_outputs, all_names, all_targets


def save_inference_results_on_disk(loader, network, name):
    logger = Logger(__name__).logger

    config = loader.config
    pack_volume = config['pack_volume']
    path = os.path.join(config['temp_folder'], name, '')
    remove_directory(path)

    network.eval()
    loss = eval(config['loss'])

    all_outputs = torch.FloatTensor().to(device=torch.device(config['device']))

    i = 1
    logger.info('Inference is in progress')
    start = datetime.datetime.now()
    logger.info('loader {}'.format(loader.batch_sampler.sampler))
    all_names = []
    all_targets = []

    total_loss = 0
    for data in tqdm(loader):
        features, lengths, metas, targets, names = data

        all_names.extend(names)
        all_targets.extend(targets)
        features = features.to(device=torch.device(config['device']))
        metas = metas.to(device=torch.device(config['device']))
        targets = targets.float().to(device=torch.device(config['device']))

        outputs = network((features, lengths, metas)).detach()

        current_loss = loss(outputs, targets, config)
        total_loss += current_loss.item()

        all_outputs = torch.cat((all_outputs, outputs.data), dim=0)

        if i % pack_volume == 0:
            filename = '%sall_outputs_%d' % (path, i)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(all_outputs, '%sall_outputs_%d' % (path, i))
            filename = '%sall_names_%d' % (path, i)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(all_names, '%sall_names_%d' % (path, i))
            filename = '%sall_targets_%d' % (path, i)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(all_targets, '%sall_targets_%d' % (path, i))

            all_outputs = torch.FloatTensor().to(device=torch.device(config['device']))
            all_names = []
            all_targets = []

        i += 1
    if loader.dataset.name == 'train':
        print(' loss ', total_loss / float(loader.__len__()))

    batches_number = len(loader) // pack_volume
    logger.info('batches_number = {}'.format(batches_number))
    end = datetime.datetime.now()
    logger.info('The inference is done in = {}'.format(end - start))
    return batches_number


def inference(loader, model, results_file_name, inbound_data_folder):
    logger = Logger(__name__).logger

    results_dir = os.path.join('%s-results' % inbound_data_folder)
    remove_directory(results_dir)
    progress_file = os.path.join(results_dir, 'progress.txt')
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)

    all_outputs, all_names, all_targets = outputs_for_large_dataset(loader, model)

    logger.info('all_names {} {} '.format(len(all_names), all_names))
    logger.info('all_outputs {} {}'.format(len(all_outputs), all_outputs[0].shape))
    logger.info('all_targets {} {}'.format(len(all_targets), all_targets[0].shape))

    write_results_in_my_format(all_names, all_outputs, results_file_name)
    return results_dir


def do_inference(config, models, results_file_name, inbound_data_folder, files_with_weights, fold_number=0):
    logger = Logger(__name__).logger
    test_data_loader = RinseDataLoader(config, name=results_file_name, inbound_folder=inbound_data_folder,
                                       fold_number=fold_number, evaluate=True)
    for model, file in zip(models, files_with_weights):
        model = load_the_model(config, file, model)

    logger.info('Do inference')
    results_file_name = 'submission_%s.csv' % results_file_name

    compound_model = CompoundModel(models)
    results_dir = inference(test_data_loader, compound_model, results_file_name, inbound_data_folder)

    return results_dir


def load_the_model(config, file_with_weights, model):
    if config['device'] == 'cpu':
        checkpoint_for_model = torch.load(file_with_weights, map_location=lambda storage, loc: storage)
    else:
        checkpoint_for_model = torch.load(file_with_weights)
    model.load_state_dict(checkpoint_for_model['state_dict'])
    model = model.to(device=torch.device(config['device']))
    model.eval()
    return model
