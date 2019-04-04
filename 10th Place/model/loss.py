import torch


def mse_loss(y_input, y_target, config):
    assert y_input.shape == y_target.shape, \
        'We need input and target of the same shape, ' \
        'currently y_input %s and y_target %s' % (y_input.shape, y_target.shape)
    loss = torch.nn.MSELoss()
    return loss(y_input, y_target.float())


def huber(y_input, y_target, config):
    assert y_input.shape == y_target.shape, \
        'We need input and target of the same shape, ' \
        'currently y_input %s and y_target %s' % (y_input.shape, y_target.shape)
    loss = torch.nn.SmoothL1Loss()
    return loss(y_input, y_target.float())


def mape_loss(y_input, y_target, config):
    threshold = 290000.0 * 0.000001
    threshold = threshold * torch.ones(y_target.shape).to(device=torch.device(config['device']))

    for_max = torch.cat((torch.abs(y_target).unsqueeze(0), threshold.unsqueeze(0)), dim=0)
    max_tensor, _ = torch.max(for_max, dim=0)
    loss = (torch.abs(y_target - y_input) / max_tensor).mean()
    return loss
