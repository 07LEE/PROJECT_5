import torch
from transformers import AutoTokenizer

from training.arguments import get_train_args
from training.bert_features import convert_examples_to_features
from training.data_prep import build_data_loader
from training.load_name_list import get_alias2id
from training.train_model import CSN, KCSN

MODEL_NAME = 'CSN'
DEVICE = 'cpu'
CHECK = 'training/test/test.txt'

if __name__ == '__main__':
    # 저장된 모델 상태 사전 로드 -------------------------------------------
    model_state_dict = torch.load("training/test_model/model.pth")
    args = get_train_args()
    name_list_path = args.name_list_path
    alias2id = get_alias2id(name_list_path)

    # 입력 데이터 준비
    check_data = build_data_loader(CHECK, alias2id, args)
    check_data_iter = iter(check_data)

    _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, _, name_list_index = next(
    check_data_iter)

    model = CSN(args)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)
    features = convert_examples_to_features(examples=CSSs, tokenizer=tokenizer)
    model.load_state_dict(model_state_dict)

    predictions = model(features, sent_char_lens,mention_poses, quote_idxes,true_index, device=DEVICE)

    scores, scores_false, scores_true = predictions

    # elif MODEL_NAME == 'KCSN':
    #     model = KCSN(args)
    #     features, tokens_list = convert_examples_to_features(CSSs, tokenizer, is_Kfeatures=True)
    #     model.load_state_dict(model_state_dict)

    # 예측 출력
    print('predictions : ', predictions)
    print('scores : ', scores)
    print('scores_false : ', scores_false)
    print('scores_true : ', scores_true)
    print('name_list_index : ', name_list_index)
