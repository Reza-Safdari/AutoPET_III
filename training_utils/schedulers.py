from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR


def build_scheduler(scheduler_name, optimizer, scheduler_steps):
    if scheduler_name.lower() == 'cosineannealinglr':
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_steps
        )
    elif scheduler_name.lower() == 'polylr':
        return PolynomialLR(
            optimizer,
            total_iters=scheduler_steps,
            power=0.9
        )
    elif scheduler_name is None:
        return None
