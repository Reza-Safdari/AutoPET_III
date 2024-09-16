import torch
from torch import nn
# import monai.transforms as mt
from monai.networks.nets import UNet, SegResNet, DynUNet, SegResNetDS, SwinUNETR, SegResNetVAE
# from monai.inferers import sliding_window_inference


# def preprocess_pred_mask(pred_mask):
#     # Define the spacing for the images
#     spacing = (2.0364201068878174, 2.03642010688781740, 3.0)
#
#     transforms = [
#         mt.EnsureChannelFirst(),
#         mt.EnsureType(),
#         mt.Orientation(axcodes="LAS"),
#         mt.Spacing(pixdim=spacing, mode="nearest"),
#         mt.EnsureType(),
#         mt.ToTensor(),
#     ]
#     transform = mt.Compose(transforms)
#     return transform(pred_mask)


class CombinedSwinUNETR(nn.Module):
    def __init__(self, in_channels_stage1, out_channels, patch_size):
        self.patch_size = patch_size
        in_channels_stage2 = in_channels_stage1 + 1

        kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
        self.dynunet = DynUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            act_name=("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
            deep_supervision=False,
            deep_supr_num=3,
            res_block=True,
        )
        checkpoint = torch.load("weights/dynunet_stage1_ckpt_00662_dice0.6657.pth", map_location=torch.device('cpu'))
        self.dynunet.load_state_dict(checkpoint["state_dict"])
        for param in self.dynunet.parameters():
            param.requires_grad = False
        # self.dynunet.eval()

        self.swinunetr_stage2 = SwinUNETR(
            img_size=patch_size,
            in_channels=in_channels_stage2,
            out_channels=out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True
        )
        # self.sw_batch_size = 4
        # self.sw_overlap = 0.5
        # self.sw_mode = "constant"

    # def dynunet_inference(self, image):
    #     return sliding_window_inference(
    #         inputs=image,
    #         roi_size=self.patch_size,
    #         sw_batch_size=self.sw_batch_size,
    #         predictor=self.dynunet,
    #         overlap=self.sw_overlap,
    #         mode=self.sw_mode,
    #     )
    #
    # def predict_mask(self, x):
    #     pred = self.dynunet_inference(x)
    #     pred = torch.sigmoid(pred)
    #     pred_mask = torch.ge(pred, 0.5)
    #     pred_mask = preprocess_pred_mask(pred_mask)
    #     return pred_mask.detach()
    #
    # def forward(self, x):
    #     self.dynunet.eval()
    #     pred_mask = self.dynunet(x)
    #     x1 = torch.cat([x, pred_mask], dim=1)
    #     x2 = self.swinunetr_stage2(x1)
    #     return x2