"""
학습한 모델에 직접 데이터를 넣어서 확인하기 위한 코드
"""
# %%
import torch
from transformers import AutoTokenizer

from arguments import get_train_args
from bert_features import convert_examples_to_features
from data_prep import build_data_loader
from load_name_list import get_alias2id
from train_model import KCSN

def _check():
    """
    결과 확인하기 위한 코드
    """
    check = 'test_model/test.txt'

    # 저장된 모델 상태 사전 로드 -------------------------------------------
    model_state_dict = torch.load("test_model/final_model.pth")
    args = get_train_args()
    name_list_path = 'data_sets/name_list.txt'
    alias2id = get_alias2id(name_list_path)

    model = KCSN(args)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)

    # 입력 데이터 준비
    check_data = build_data_loader(check, alias2id, args)
    check_data_iter = iter(check_data)

    for _ in range(len(check_data)):
        _, css, scl, mp, qi, cc, _, ti, _, name_list_index = next(check_data_iter)
        features, tokens_list = convert_examples_to_features(examples=css, tokenizer=tokenizer)
        model.load_state_dict(model_state_dict)

        try:
            predictions = model(features, scl, mp, qi, ti, "cuda:0", tokens_list, cc)
        except RuntimeError:
            predictions = model(features, scl, mp, qi, ti, "cpu", tokens_list, cc)

        # scores, scores_false, scores_true = predictions
        scores, _, _ = predictions

        # 예측 출력
        # print('----------------------------------------------------------------')
        # print('predictions: ', predictions)
        # print('scores: ', list(scores))
        # print('scores_false: ', scores_false)
        # print('scores_true: ', scores_true)
        # print('name_list_index: ', name_list_index)
        # print('----------------------------------------------------------------')

        # 후처리
        correct = 'True' if scores.max(0)[1].item() == ti else "False"
        scores_np = scores.detach().cpu().numpy()
        scores_list = scores_np.tolist()
        score_index = scores_list.index(max(scores_list))
        name_index = name_list_index[score_index]

        for key, val in alias2id.items():
            if val == name_index:
                result_key = key

        print(f'{correct}: ({name_index}) {result_key}')

if __name__ == '__main__':
    _check()

# %%
