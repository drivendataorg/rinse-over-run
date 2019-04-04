import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from logger import Logger


class RinseDataset(Dataset):
    def __init__(self, config, name, evaluate=False, inbound_folder='', fold_number=0, evaluate_on_train=False):
        if inbound_folder != '':
            self.data_folder = inbound_folder
        else:
            self.data_folder = config['data_loader']['data_dir_%s' % name]
        self.name = name
        self.fold_number = fold_number
        self.config = config
        self.evaluate = evaluate
        self.logger = Logger(__name__).logger
        self.evaluate_on_train = evaluate_on_train
        self.ts_cols_numeric = [
            'timestamp',

            'supply_flow',
            'supply_pressure',
            'return_temperature',
            'return_conductivity',
            'return_turbidity',
            'return_flow',
            'tank_level_pre_rinse',
            'tank_level_caustic',
            'tank_level_acid',
            'tank_level_clean_water',
            'tank_temperature_pre_rinse',
            'tank_temperature_caustic',
            'tank_temperature_acid',
            'tank_concentration_caustic',
            'tank_concentration_acid', ]

        self.ts_cols_bool = [
            'supply_pump',
            'supply_pre_rinse',
            'supply_caustic',
            'return_caustic',
            'supply_acid',
            'return_acid',
            'supply_clean_water',
            'return_recovery_water',
            'return_drain',
            'object_low_level',
            'tank_lsh_caustic',
            'tank_lsh_clean_water'

        ]
        self.ts_cols = self.ts_cols_numeric + self.ts_cols_bool
        self.recipe_cols = ['pre_rinse', 'caustic', 'intermediate_rinse', 'acid', 'final_rinse']
        self.tf_pipeline_cols = [
            'pipeline_L1',
            'pipeline_L2',
            'pipeline_L3',
            'pipeline_L4',

            'pipeline_L6',
            'pipeline_L7',
            'pipeline_L8',
            'pipeline_L9',
            'pipeline_L10',
            'pipeline_L11',
            'pipeline_L12', ]

        self.tf_objects_cols = [
            'object_id_102', 'object_id_103', 'object_id_107', 'object_id_108', 'object_id_109', 'object_id_110',
            'object_id_111', 'object_id_112', 'object_id_113', 'object_id_114', 'object_id_115', 'object_id_116',
            'object_id_204', 'object_id_205', 'object_id_210', 'object_id_211', 'object_id_212', 'object_id_213',
            'object_id_214', 'object_id_215', 'object_id_216', 'object_id_217', 'object_id_300', 'object_id_301',
            'object_id_302', 'object_id_303', 'object_id_304', 'object_id_305', 'object_id_306', 'object_id_308',
            'object_id_309', 'object_id_405', 'object_id_409', 'object_id_420', 'object_id_421', 'object_id_424',
            'object_id_426', 'object_id_427', 'object_id_428', 'object_id_429', 'object_id_431', 'object_id_434',
            'object_id_435', 'object_id_436', 'object_id_437', 'object_id_438', 'object_id_911', 'object_id_912',
            'object_id_913', 'object_id_914', 'object_id_917', 'object_id_918', 'object_id_919', 'object_id_921',
            'object_id_922', 'object_id_923', 'object_id_924', 'object_id_925', 'object_id_926', 'object_id_930',
            'object_id_932', 'object_id_933', 'object_id_934', 'object_id_938', 'object_id_940', 'object_id_941',
            'object_id_942', 'object_id_943', 'object_id_944', 'object_id_945', 'object_id_946', 'object_id_950',
            'object_id_951', 'object_id_952', 'object_id_953', 'object_id_954', 'object_id_955', 'object_id_956',
            'object_id_957', 'object_id_958', 'object_id_959', 'object_id_960', 'object_id_961', 'object_id_962',
            'object_id_963', 'object_id_964', 'object_id_965', 'object_id_966', 'object_id_910', 'object_id_920',
            'object_id_970', 'object_id_971', 'object_id_976', 'object_id_977']

        self.features, self.metas, self.targets, self.names = self.get_features_and_targets_list()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature, target = self.preprocessing_for_data_with_index(index)
        return feature, self.metas[index], target, self.names[index]

    def get_features_and_targets_list(self):
        features, metas, targets, names = self.get_all_data_lists()

        if self.name == 'train':
            train_features, train_metas, train_new_targets, train_names, val_features, val_metas, val_new_targets, val_names = \
                self.select_fold(features, metas, targets, names)
            if self.evaluate:
                return val_features, val_metas, val_new_targets, val_names
            else:
                return train_features, train_metas, train_new_targets, train_names
        else:
            self.logger.info('names {}'.format(names))
            return features, metas, targets, names

    def select_fold(self, features, metas, new_targets, names):
        total_features = len(features)
        indices = np.arange(start=0, stop=total_features)
        np.random.seed(seed=1986)
        shuffled_indices = np.random.permutation(indices)
        n_splits = self.config['n_splits']
        fold_size = total_features // n_splits
        folds = []
        for i in range(n_splits):
            if i < n_splits - 1:
                folds.append(shuffled_indices[i * fold_size: (i + 1) * fold_size])
            else:
                folds.append(shuffled_indices[i * fold_size:])
        train_index = np.array([], dtype=int)
        for fold_number, fold in enumerate(folds):
            print('fold ', fold.shape, end='')
            if fold_number != self.fold_number:
                train_index = np.hstack((train_index, fold))
            else:
                validation_index = fold

        features = np.array(features)
        metas = np.array(metas)
        new_targets = np.array(new_targets)
        names = np.array(names)

        return features[train_index], metas[train_index], new_targets[train_index], names[train_index], \
               features[validation_index], metas[validation_index], new_targets[validation_index], names[
                   validation_index]

    def preprocessing_for_data_with_index(self, index):
        feature = self.features[index]
        target = self.targets[index]

        return feature, target

    def get_all_data_lists(self):
        features = []
        metas = []
        targets = []
        names = []
        if self.name == 'train' or (self.name == 'test' and self.evaluate_on_train):
            # for training our model
            values = pd.read_csv(os.path.join(self.data_folder, 'train_values.csv'),
                                 index_col=0,
                                 parse_dates=['timestamp'])
        else:
            # for testing our model
            values = pd.read_csv(os.path.join(self.data_folder, 'test_values.csv'),
                                 index_col=0,
                                 parse_dates=['timestamp'])

        recipes = pd.read_csv(os.path.join(self.data_folder, 'recipe_metadata.csv'))

        values[self.ts_cols_bool] = values[self.ts_cols_bool].astype(int)

        values['timestamp'] = values['timestamp'].dt.dayofweek

        values = pd.concat([values,
                            pd.get_dummies(values['pipeline'],
                                           prefix='pipeline')],
                           axis=1).drop(['pipeline'], axis=1)

        values = pd.concat([values,
                            pd.get_dummies(values['object_id'],
                                           prefix='object_id')],
                           axis=1).drop(['object_id'], axis=1)

        if self.name == 'test' and (not self.evaluate_on_train):
            values['pipeline_L12'] = np.zeros((len(values)), dtype=int)
            # objects_ids  910, 920, 970, 971, 976, 977
            for obj_id in [910, 920, 970, 971, 976, 977]:
                values['object_id_%d' % obj_id] = np.zeros((len(values)), dtype=int)

        # supply_flow        103161
        # supply_pressure        6.19401
        # return_temperature        96.9763
        # return_conductivity        73.5659
        # return_turbidity        100.966
        # return_flow        103139
        # tank_level_pre_rinse        58.3491
        # tank_level_caustic        51.3549
        # tank_level_acid        52.441
        # tank_level_clean_water        50.6355
        # tank_temperature_pre_rinse        37.7894
        # tank_temperature_caustic        83.5286
        # tank_temperature_acid        73.9945
        # tank_concentration_caustic        61.4609
        # tank_concentration_acid        65.2409

        if self.name == 'train' or (self.name == 'test' and self.evaluate_on_train):
            final_phases = values[values.target_time_period]

        values['supply_flow'] = values['supply_flow'] * 0.0001
        values['return_flow'] = values['return_flow'] * 0.0001
        values['return_turbidity'] = values['return_turbidity'] * 0.1

        # subset to final rinse phase observations
        pre_rinses = values[values.phase == 'pre_rinse']
        caustics = values[values.phase == 'caustic']
        intermediate_rinses = values[values.phase == 'intermediate_rinse']
        acids = values[values.phase == 'acid']

        all_processes_ids = list(values.groupby('process_id').process_id.nunique().index)

        print('len(all_processes_ids) ', len(all_processes_ids))

        numerical_data_scale_coefficient = 0.01
        for process_id in all_processes_ids:
            # let's look at just one process

            pre_rinse = np.array(pre_rinses[pre_rinses.process_id == process_id][self.ts_cols]) \
                        * numerical_data_scale_coefficient
            pre_rinse = np.hstack((pre_rinse, 1.0 * np.ones((pre_rinse.shape[0], 1))))

            caustic = np.array(caustics[caustics.process_id == process_id][self.ts_cols]) \
                      * numerical_data_scale_coefficient
            caustic = np.hstack((caustic, 2.0 * np.ones((caustic.shape[0], 1))))

            intermediate_rinse = np.array(
                intermediate_rinses[intermediate_rinses.process_id == process_id][self.ts_cols]) \
                                 * numerical_data_scale_coefficient
            intermediate_rinse = np.hstack((intermediate_rinse, 3.0 * np.ones((intermediate_rinse.shape[0], 1))))

            acid = np.array(acids[acids.process_id == process_id][self.ts_cols]) \
                   * numerical_data_scale_coefficient
            acid = np.hstack((acid, 4.0 * np.ones((acid.shape[0], 1))))

            recipe = np.array(recipes[recipes.process_id == process_id][self.recipe_cols]) \
                     * numerical_data_scale_coefficient
            pipelines_data = np.array(values[values.process_id == process_id][self.tf_pipeline_cols])[0].reshape(1, -1) \
                             * numerical_data_scale_coefficient
            objects_data = np.array(values[values.process_id == process_id][self.tf_objects_cols])[0].reshape(1, -1) \
                           * numerical_data_scale_coefficient

            sequence = np.vstack((pre_rinse, caustic, intermediate_rinse, acid))
            sequence_1 = pre_rinse
            sequence_2 = np.vstack((pre_rinse, caustic))
            sequence_3 = np.vstack((pre_rinse, caustic, intermediate_rinse))

            meta = np.hstack((recipe, pipelines_data, objects_data))

            target = 0.0
            if self.name == 'train' or (self.name == 'test' and self.evaluate_on_train):
                final_phase = final_phases[final_phases.process_id == process_id]
                # calculate target variable
                final_phase = final_phase.assign(
                    target=np.maximum(final_phase.return_flow, 0) * final_phase.return_turbidity
                )
                target = final_phase.target.sum() * 0.000001

            # the process is useful if only it contains something but the final phase data
            if sequence.shape[0] != 0:
                features.append(sequence)
                metas.append(meta)
                targets.append(target)
                names.append(process_id)

            if self.name == 'train' or \
                    (self.name == 'test' and self.evaluate_on_train) or \
                    (self.name == 'test' and self.config['tta']):
                self.augment_with_part_crops(features, names, process_id, sequence_1, sequence_2, sequence_3,
                                             meta, metas, target, targets)

                # augment with gaussian noise
                self.augment_with_gaussian_noise(features, names, process_id, sequence, sequence_1, sequence_2,
                                                 sequence_3, meta, metas, target, targets)

        return features, metas, targets, names

    @staticmethod
    def augment_with_part_crops(features, names, process_id, sequence_1, sequence_2, sequence_3, meta, metas, target,
                                targets):
        if sequence_1.shape[0] != 0:
            features.append(sequence_1)
            metas.append(meta)
            targets.append(target)
            names.append(process_id + 1000000)
        if sequence_2.shape[0] != 0:
            features.append(sequence_2)
            metas.append(meta)
            targets.append(target)
            names.append(process_id + 2000000)
        if sequence_3.shape[0] != 0:
            features.append(sequence_3)
            metas.append(meta)
            targets.append(target)
            names.append(process_id + 3000000)

    def augment_with_gaussian_noise(self, features, names, process_id, sequence, sequence_1, sequence_2, sequence_3,
                                    meta, metas, target, targets):
        noise_scale = 0.001
        if sequence.shape[0] != 0:
            features.append(sequence +
                            noise_scale * np.random.randn(sequence.shape[0],
                                                          sequence.shape[1]).astype('float32'))

            metas.append(meta)
            targets.append(target)
            names.append(process_id + 4000000)
        if sequence_1.shape[0] != 0:
            features.append(sequence_1 +
                            noise_scale * np.random.randn(sequence_1.shape[0],
                                                          sequence_1.shape[1]).astype('float32'))
            metas.append(meta)
            targets.append(target)
            names.append(process_id + 5000000)

        if sequence_2.shape[0] != 0:
            features.append(sequence_2 +
                            noise_scale * np.random.randn(sequence_2.shape[0],
                                                          sequence_2.shape[1]).astype('float32'))
            metas.append(meta)
            targets.append(target)
            names.append(process_id + 6000000)

        if sequence_3.shape[0] != 0:
            features.append(sequence_3 +
                            noise_scale * np.random.randn(sequence_3.shape[0],
                                                          sequence_3.shape[1]).astype('float32'))
            metas.append(meta)
            targets.append(target)
            names.append(process_id + 7000000)
