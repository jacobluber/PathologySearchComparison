import torch
import torch.nn as nn
import ResNet as ResNet

def ccl_model(checkpoint_path=r'./checkpoints/best_ckpt.pth'):
    model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
    pretext_model = torch.load(checkpoint_path)
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    return model