"""
모델들 불러오는 모듈
"""
import torch

def load_ner(path ='model/NER.pth'):
    """
    NER 모델
    """
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint

def load_fs(path = 'model/FS.pth'):
    """
    Find Speaker 모델
    """
    model = torch.load(path)
    return model
