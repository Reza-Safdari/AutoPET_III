from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from torch import nn


class Loss(nn.Module):
    def __init__(self, loss_name, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        if loss_name.lower() == 'diceceloss':
            self.loss_fn = DiceCELoss(
                sigmoid=True,
                batch=True,
                include_background=True
            )

        elif loss_name.lower() == 'dicefocalloss':
            self.loss_fn = DiceFocalLoss(
                include_background=True,
                sigmoid=True,
                batch=True,
                squared_pred=True,
            )
        elif loss_name.lower() == 'diceloss':
            self.loss_fn = DiceLoss(
                to_onehot_y=True,
                softmax=True
            )
        else:
            raise NotImplementedError(f"Loss {loss_name} not implemented")

    def compute_loss(self, prediction, label):
        if self.deep_supervision:
            loss, weights = 0.0, 0.0
            for i in range(prediction.shape[1]):
                loss += self.loss_fn(prediction[:, i], label) * 0.5 ** i
                weights += 0.5 ** i
            return loss / weights
        return self.loss_fn(prediction, label)
