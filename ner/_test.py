"""
NER 모델 확인
"""
# %%
import torch

from ner_utils import show_tokens

def check(texts):
    """
    모델이 잘 저장되고 불러와지는지 확인을 하는 코드.
    """
    models ='test/ner_model.pth'

    # 모델 불러오기
    checkpoint = torch.load(models)
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']
    # optimizer = checkpoint['optimizer_state_dict']

    # 기타 정보
    # args = checkpoint['args']
    # unique_tags = checkpoint['unique_tags']
    tag2id = checkpoint['tag2id']
    # id2tag = checkpoint['id2tag']

    # 모델 상태 적용
    model.load_state_dict(checkpoint['model_state_dict'])

    # 디바이스에 할당 (GPU or CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    show_tokens(texts, model, device, tokenizer, tag2id)

if __name__ == '__main__':
    TEXT = '처음 몇 달간은 그랬다.'
    check(TEXT)
