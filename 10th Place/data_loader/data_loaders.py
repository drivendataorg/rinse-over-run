import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import numpy as np
from datasets import rinse_dataset


class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [torch.from_numpy(x[0]) for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
        # Don't forget to grab the metas, labels and names of the *sorted* batch
        metas = torch.FloatTensor(np.array(list(map(lambda x: x[1], sorted_batch))))
        labels = torch.FloatTensor(np.array(list(map(lambda x: x[2], sorted_batch))))
        names = torch.LongTensor(np.array(list(map(lambda x: x[3], sorted_batch))))
        return sequences_padded, lengths, metas, labels, names


class RinseDataLoader(DataLoader):
    """
    Data loading
    """

    def __init__(self, config, name, inbound_folder, fold_number, evaluate=False, evaluate_on_train=False):
        super(RinseDataLoader, self).__init__(
            dataset=rinse_dataset.RinseDataset(config=config, name=name,
                                               inbound_folder=inbound_folder, evaluate=evaluate,
                                               fold_number=fold_number,
                                               evaluate_on_train=evaluate_on_train),
            batch_size=config['data_loader']['evaluation_batch_size'] if evaluate else config['data_loader'][
                'batch_size_%s' % name],
            shuffle=False if evaluate else config['data_loader']['shuffle'],
            drop_last=config['data_loader']['drop_last'],
            collate_fn=PadSequence()
        )

        self.batch_sampler = BatchSampler(
            SequentialSampler(self.dataset),
            batch_size=self.batch_size,
            drop_last=self.drop_last
        )

        self.config = config
