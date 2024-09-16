from torch.optim import Adam, SGD
from monai.optimizers import Novograd


def build_optimizer(optim_name, params, lr, weight_decay, momentum, nesterov):
    if optim_name.lower() == 'sgd':
        return SGD(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optim_name.lower() == 'adam':
        return Adam(
            params,
            lr=lr
        )
    elif optim_name.lower() == 'novograd':
        return Novograd(
            params,
            lr=lr
        )
    elif optim_name.lower() == 'adamw':
        return Adam(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f'Optimizer {optim_name} not supported')
