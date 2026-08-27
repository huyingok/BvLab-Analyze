import torch
from CellUnet3D_DDP.models.UnetModel_V2 import UNet3D
# from models.MyModelUnet1 import UNet3D


def LoadModel(model_config, modelPath=None):
    model = UNet3D(**model_config)
    if not modelPath is None:
        ckpt = torch.load(modelPath, map_location='cuda:0', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
    return model
