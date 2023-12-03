import torch

from training.arguments import get_train_args
from training.bert_features import convert_examples_to_features
from training.bert_features import convert_examples_to_Kfeatures
from training.data_prep import build_data_loader
from training.load_name_list import get_alias2id
from training.train_model import CSN, KCSN

MODEL_NAME = 'CSN'

if __name__ == '__main__':
    # 저장된 모델 상태 사전 로드 -------------------------------------------
    model_state_dict = torch.load("final_model.pth")
    args = get_train_args()
    name_list_path = args.name_list_path
    alias2id = get_alias2id(name_list_path)

    if MODEL_NAME == 'CSN':
        model = CSN(args)
        features = convert_examples_to_features(examples=CSSs, tokenizer=tokenizer)
    else:
        model = KCSN(args)

    # 새로운 모델 인스턴스 생성 및 상태 사전 로드
    model.load_state_dict(model_state_dict)

    # 입력 데이터 준비
    check = 'training/test/test.txt'
    check_data = build_data_loader(check, alias2id, args)
    check_data_iter = iter(check_data)

    _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, _ = next(
        check_data_iter)

    # 예측 수행
    predictions = model(features=features, sent_char_lens=sent_char_lens,
                        mention_poses=mention_poses, quote_idxes=quote_idxes,
                        true_index=true_index, device='cpu')
    scores, scores_false, scores_true = predictions

    # 예측 출력
    print('predictions : ', predictions)
    print('scores : ', scores)
    print('scores_false : ', scores_false)
    print('scores_true : ', scores_true)
