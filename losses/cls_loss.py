# losses/cls_loss.py
import torch.nn as nn

def get_cls_loss():
    """
    Returns the loss function for the classification task.
    """
    return nn.CrossEntropyLoss()
