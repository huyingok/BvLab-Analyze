import torch.nn as nn
from Unet3D.pytorch3dunet.unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders, Encoder, SingleConv
from Unet3D.pytorch3dunet.unet3d.utils import get_class, number_of_features_per_level
import torch

class EncoderZ(nn.Module):
    def __init__(self, input, output, encoder=True, kernel_size=3, order='gcr',
                   num_groups=8, padding=1, upscale=2, dropout_prob=0.1,
                   is3d=True):
        super(EncoderZ, self).__init__()
        self.MaxResizeZ = nn.MaxPool3d(kernel_size=(2, 1, 1))
        self.Conv = DoubleConv(input, output, encoder=encoder, kernel_size=kernel_size, order=order,
                   num_groups=num_groups, padding=padding, upscale=upscale, dropout_prob=dropout_prob,
                   is3d=is3d)

    def forward(self, x):
        x = self.MaxResizeZ(x)
        x = self.Conv(x)
        return x

class DoubleConvGN(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        dropout_prob (float or tuple): dropout probability for each convolution, default 0.1
        is3d (bool): if True use Conv3d instead of Conv2d layers
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr',
                 num_groups=[8, 8], padding=1, upscale=2, dropout_prob=0.1, is3d=True):
        super(DoubleConvGN, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            if upscale == 1:
              conv1_out_channels = out_channels
            else:
              conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # check if dropout_prob is a tuple and if so
        # split it for different dropout probabilities for each convolution.
        if isinstance(dropout_prob, list) or isinstance(dropout_prob, tuple):
          dropout_prob1 = dropout_prob[0]
          dropout_prob2 = dropout_prob[1]
        else:
          dropout_prob1 = dropout_prob2 = dropout_prob

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups[0],
                                   padding=padding, dropout_prob=dropout_prob1, is3d=is3d))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups[1],
                                   padding=padding, dropout_prob=dropout_prob2, is3d=is3d))

class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """
    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, fieldSpace=2, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, is3d=True):
        super(AbstractUNet, self).__init__()
        self.in_channels = in_channels
        self.ours_channels = self.in_channels * 2
        self.input2 = None
        self.fieldSpace = fieldSpace
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"
        conv_upscale = 2
        dropout_prob = 0.1
        self.AvgResizeXY = nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.upsamplingEnd = nn.ConvTranspose3d(f_maps[0], int(f_maps[0] // 2), kernel_size=conv_kernel_size,
                                                stride=(1, 2, 2), padding=1, bias=False)
        # Encoder
        self.Encoder_1 = [
            DoubleConvGN(4, f_maps[0], encoder=True,
                order=layer_order, kernel_size=conv_kernel_size, num_groups=[8, 8], padding=conv_padding,
                upscale=conv_upscale, dropout_prob=dropout_prob, is3d=is3d),
            EncoderZ(f_maps[0], f_maps[1], encoder=True, kernel_size=conv_kernel_size, order=layer_order,
                     num_groups=num_groups, padding=conv_padding, upscale=conv_upscale, dropout_prob=dropout_prob,
                     is3d=is3d)
        ]
        for fi in range(2, len(f_maps)):
            self.Encoder_1.append(
                Encoder(f_maps[fi - 1], f_maps[fi], basic_module=basic_module, conv_layer_order=layer_order, conv_kernel_size=conv_kernel_size,
                        num_groups=num_groups, pool_kernel_size=pool_kernel_size, padding=conv_padding, upscale=conv_upscale,
                        dropout_prob=dropout_prob, is3d=is3d)
            )
        self.Encoder_1 = nn.ModuleList(self.Encoder_1)
        self.Encoder_2 = [
            DoubleConvGN(self.ours_channels, f_maps[0], encoder=True, order=layer_order, kernel_size=conv_kernel_size,
                num_groups=[8, 8], padding=conv_padding, upscale=conv_upscale,
                         dropout_prob=dropout_prob, is3d=is3d),
            EncoderZ(f_maps[0], f_maps[1], encoder=True, kernel_size=conv_kernel_size, order=layer_order,
                     num_groups=num_groups, padding=conv_padding, upscale=conv_upscale, dropout_prob=dropout_prob,
                     is3d=is3d)
        ]
        for fi in range(2, len(f_maps)):
            self.Encoder_2.append(
                Encoder(f_maps[fi - 1], f_maps[fi], basic_module=basic_module, conv_layer_order=layer_order, conv_kernel_size=conv_kernel_size,
                        num_groups=num_groups, pool_kernel_size=pool_kernel_size, padding=conv_padding, upscale=conv_upscale,
                        dropout_prob=dropout_prob, is3d=is3d)
            )
        self.Encoder_2 = nn.ModuleList(self.Encoder_2)
        # Decoder
        upsample = 'default'
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, upsample, dropout_prob, is3d)
        self.addDec = nn.ModuleList()
        for f in f_maps[-1::-1]:
            self.addDec.append(DoubleConv(f * 2, f, encoder=False, kernel_size=conv_kernel_size, order=layer_order,
                                          num_groups=num_groups, padding=conv_padding, is3d=is3d))
        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(int(f_maps[0] // 2), out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(int(f_maps[0] // 2), out_channels, 1)
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        oriShape = x.shape
        xx = self.AvgResizeXY(x)
        # xx = x
        if self.input2 is None or self.input2.shape[-3:] != xx.shape[-3:] or self.input2.shape[0] != xx.shape[0]:
            batchSize = xx.size(0)
            imgSize = xx.shape[-3:]
            segLen = self.in_channels * self.fieldSpace
            self.sInd = segLen // 2 - 1
            self.xLen = imgSize[2] - segLen + self.fieldSpace
            self.yLen = imgSize[1] - segLen + self.fieldSpace
            self.input2 = torch.zeros([batchSize, self.ours_channels, *imgSize], dtype=torch.float32, device=x.device)
            # x轴
        # x轴
        for i in range(0, self.in_channels):
            j = i * self.fieldSpace
            self.input2[:, i, :, :, self.sInd: self.sInd + self.xLen] = xx[:, 0, :, :, j: j + self.xLen]
            self.input2[:, i, :, :, :self.sInd] = self.input2[:, i, :, :, self.sInd, None]
            self.input2[:, i, :, :, self.sInd + self.xLen:] = self.input2[:, i, :, :, self.sInd + self.xLen - 1, None]
        # y轴
        for i in range(0, self.in_channels):
            j = i * self.fieldSpace
            i2 = self.in_channels + i
            self.input2[:, i2, :, self.sInd: self.sInd + self.yLen] = xx[:, 0, :, j: j + self.yLen]
            self.input2[:, i2, :, :self.sInd] = self.input2[:, i2, :, None, self.sInd]
            self.input2[:, i2, :, self.sInd + self.yLen:] = self.input2[:, i2, :, self.sInd + self.yLen - 1, None]
        xx = self.input2 / (xx + 0.01)
        x = torch.cat([x[..., ::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2], x[..., 1::2, ::2]], dim=1)
        encoders_features = []
        for encoder1, encoder2, addDec in zip(self.Encoder_1, self.Encoder_2, self.addDec[::-1]):
            x = encoder1(x)
            xx = encoder2(xx)
            encoders_features.insert(0, addDec(torch.cat([x, xx], dim=1)))
        x = encoders_features.pop(0)
        for di, decoder in enumerate(self.decoders):
            encoders_feature = encoders_features.pop(0)
            x = decoder(encoders_feature, x)
        x = self.upsamplingEnd(x, oriShape[2:])
        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        return x

class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, fieldSpace=2, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     fieldSpace=fieldSpace,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     is3d=True)

class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             is3d=True)

class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               final_sigmoid=final_sigmoid,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               is_segmentation=is_segmentation,
                                               conv_padding=conv_padding,
                                               is3d=True)

class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     is3d=False)

def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
        'pytorch3dunet.unet3d.model'
    ])
    return model_class(**model_config)

if __name__ == '__main__':
    modelCfg = {
        'in_channels': 8,
        'out_channels': 1,
        'fieldSpace': 1,
        # number of features at each level of the U-Net
        'f_maps': [32, 64, 128, 256, 512],
        'final_sigmoid': True,
    }
    device = torch.device('cuda:0')
    obj1 = UNet3D(**modelCfg).to(device)
    images = torch.zeros([1, 1, 128, 128, 128], dtype=torch.float32).to(device)
    out = obj1(images)
    print(out.shape)
