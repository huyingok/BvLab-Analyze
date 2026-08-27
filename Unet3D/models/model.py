import torch
# from models.UnetModel5 import UNet3D
from Unet3D.models.Unet3D_V3 import UNet3D


def LoadModel(model_config, modelPath=None):
    model = UNet3D(**model_config)
    if not modelPath is None:
        ckpt = torch.load(modelPath, map_location='cpu', weights_only=False)
        # ckpt = torch.load(modelPath)
        model.load_state_dict(ckpt['state_dict'])
    return model

