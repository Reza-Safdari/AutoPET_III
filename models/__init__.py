from torch import nn
# from models.MirrorUNet import Mirror_UNet
from monai.networks.nets import UNet, SegResNet, DynUNet, SegResNetDS, SwinUNETR, SegResNetVAE
from monai.networks.layers.factories import Norm, Act

# from .dynunet import DynUNet
from .UxLSTMEnc_3d import UXlstmEnc
from .attention_models import AttentionUnet, AttentionSegResNet
from .combined_models import CombinedSwinUNETR


def build_model(model_name, dataset_name, patch_size=(128, 160, 112), use_cascading=False, deep_supervision=False):
    in_channels = 2 if dataset_name.lower() == 'autopet' else 1
    in_channels = 3 if use_cascading else in_channels
    if model_name.lower() == 'unet':
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,
            channels=(32, 64, 128, 256, 512) if dataset_name.lower() == 'autopet' else (16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            dropout=0.2 if dataset_name.lower() == 'autopet' else 0.0,
            bias=True)
    elif model_name.lower() == 'segresnet':
        return SegResNetDS(
            spatial_dims=3,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dsdepth=3,
            init_filters=16,
            in_channels=in_channels,
            out_channels=1,
        )
    # elif model_name.lower() == 'mirrorunet':
    #     return Mirror_UNet(
    #         spatial_dims=3,
    #         in_channels=in_channels,  # must be left at 1, this refers to the #c of each individual branch (PET or CT)
    #         # out_channels=out_channels,
    #         channels=(16, 32, 64, 128, 256),
    #         strides=(2, 2, 2, 2),
    #         num_res_units=2,
    #         norm=Norm.BATCH,
    #         task="args.task",
    #         args="args")
    elif model_name.lower() == "dynunet":
        kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
        return DynUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            act_name=("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
            deep_supervision=deep_supervision,
            deep_supr_num=3,
            res_block=True,
        )
    elif model_name.lower() == "uxlstmenc_3d":
        kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 1]]
        return UXlstmEnc(
            input_size=patch_size,
            input_channels=2,
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 280, 280],
            conv_op=nn.Conv3d,
            kernel_sizes=kernels,
            strides=strides,
            n_conv_per_stage=[2, 2, 2, 2, 2, 2],
            num_classes=1,
            n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=deep_supervision,
        )
    elif model_name.lower() == "swinunetr":
        return SwinUNETR(
            img_size=patch_size,
            in_channels=in_channels,
            out_channels=1,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True
        )
    elif model_name.lower() == "segresnetvae":
        return SegResNetVAE(
            input_image_size=patch_size,
            spatial_dims=3,
            blocks_down=(1, 2, 2, 4, 4, 4),
            blocks_up=(1, 1, 1, 1, 1),
            init_filters=8,
            in_channels=in_channels,
            out_channels=1,
            act=Act.RELU,
            dropout_prob=0.2,
            upsample_mode="deconv",
        )
    elif model_name.lower() == "attentionunet":
        return AttentionUnet(in_channels)
    elif model_name.lower() == "attentionsegresnet":
        return AttentionSegResNet(in_channels)
    elif model_name.lower() == "combinedswinunetr":
        return CombinedSwinUNETR(in_channels, 1, patch_size)

    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
