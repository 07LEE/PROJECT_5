"""
Model 정의
"""
from typing import Tuple, Dict, Any
import torch
from torch import nn

def load_ner_model(path ='ner/test/cpu.pth') -> Tuple[nn.Module, Dict[str, Any]]:
    """
    NER 모델 불러오기
    """
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint
