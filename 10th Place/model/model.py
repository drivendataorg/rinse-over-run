import torch
from torch import nn

from base import BaseModel


class LSTM_and_4_linear(nn.Module):
    def __init__(self, config, input_size=1, hidden_size=128, num_layers=3, bidirectional=True):
        super(LSTM_and_4_linear, self).__init__()
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.lstm_2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional,
                              dropout=0.5
                              )
        # recipe + pipeline data + object_id_data
        meta_data_size = 5 + 11 + 94
        linear_input_size = self.num_directions * hidden_size + meta_data_size
        intermediate_number_of_features = linear_input_size // 2
        self.linear_1 = nn.Sequential(
            nn.Linear(in_features=linear_input_size, out_features=intermediate_number_of_features),
            nn.BatchNorm1d(num_features=intermediate_number_of_features),
            nn.ReLU(inplace=True)
        )
        next_intermediate_number_of_features = intermediate_number_of_features // 2
        self.linear_2 = nn.Sequential(nn.Linear(in_features=intermediate_number_of_features,
                                                out_features=next_intermediate_number_of_features),
                                      nn.BatchNorm1d(num_features=next_intermediate_number_of_features),
                                      nn.ReLU(inplace=True)
                                      )
        nextnext_intermediate_number_of_features = next_intermediate_number_of_features // 2
        self.linear_3 = nn.Sequential(nn.Linear(in_features=next_intermediate_number_of_features,
                                                out_features=nextnext_intermediate_number_of_features),
                                      nn.BatchNorm1d(num_features=nextnext_intermediate_number_of_features),
                                      nn.ReLU(inplace=True)
                                      )
        self.linear_4 = nn.Sequential(nn.Linear(in_features=nextnext_intermediate_number_of_features,
                                                out_features=1),
                                      )
        self.config = config

    #  **input** of shape `(batch, seq_len, input_size)` because we set batch_first=True
    def forward(self, input):
        # **output** of shape `(batch, seq_len, num_directions * hidden_size)`: tensor
        #           containing the output features `(h_t)` from the last layer of the LSTM,
        #           for each t.
        # num_directions = 1 for unidirectional LSTM or 2 for bidirectional

        input, input_lengths, meta = input

        lstm_output_2, _ = self.lstm_2(input.float())

        input_lengths = input_lengths.to(device=self.config['device']) - 1
        lstm_output_last_2 = lstm_output_2[range(input_lengths.shape[0]), input_lengths, :]

        lstm_output_last = torch.cat((lstm_output_last_2, meta.squeeze(dim=1)), dim=1)

        output = self.linear_1(lstm_output_last)
        output = self.linear_2(output)
        output = self.linear_3(output)
        output = self.linear_4(output)

        output = output.squeeze(dim=1)

        return output


class RinseModel(BaseModel):
    def __init__(self, config, model_name='GRU_and_3_linear'):
        super(RinseModel, self).__init__(config)
        self.config = config
        # phase number + numerical + boolean
        input_size = 1 + 16 + 12
        if model_name == 'LSTM_and_4_linear':
            self.net = LSTM_and_4_linear(
                config=self.config,
                input_size=input_size,
                hidden_size=input_size * 2,
                num_layers=2,
                bidirectional=False
            ).to(device=torch.device(self.config['device']))
        else:
            print('You should use LSTM_and_4_linear model!')

    def forward(self, x):
        #  **input** of shape `(batch, seq_len, input_size)` because we set batch_first=True
        x = self.net(x)
        return x


class CompoundModel(BaseModel):
    def __init__(self, models):
        super(CompoundModel, self).__init__(models)

        self.models = models

    def forward(self, x):
        model = self.models[0]
        shape = model(x).shape
        config = model.config
        summ = torch.zeros(shape).float().to(device=config['device'])

        for model in self.models:
            summ = summ.float() + model(x)
        summ = summ / float(len(self.models))

        return summ
