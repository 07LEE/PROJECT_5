import torch
from transformers import AutoTokenizer

from ..module.input_process import make_instance_list, input_data_loader, making_script
from ..model.find_speaker.load_name_list import get_alias2id
from ..model.find_speaker.arguments import get_train_args
from ..model.find_speaker.bert_features import convert_examples_to_features
from ..model.find_speaker.train_model import KCSN

from ..model.ner.ner_model import load_ner_model
# from ..model.ner.ner_utils import show_tokens

def user_input(text):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # NER_Model
    model_n, checkpoint  = load_ner_model(path='../model/NER.pth')
    model_n.to(device)
    alias2id = get_alias2id('./test/name.txt')

    # SF_Model
    model_state_dict = torch.load("../model/FS.pth")
    args_s = get_train_args()
    model_s = KCSN(args_s)
    tokenizer_s = AutoTokenizer.from_pretrained(args_s.bert_pretrained_dir)

    # # Load test Data
    # with open('./test/test.txt', 'r', encoding='utf-8') as file:
    #     input_texts = file.read()

    # data = make_ner_input(input_texts)

    # for i in data:
    #     show_tokens(text=i, checkpoint=checkpoint)

    ins_list, ins_num = make_instance_list(text)
    user_input = input_data_loader(ins_list, alias2id=alias2id)
    user_input_iter = iter(user_input)

    who = []
    for i, _ in enumerate(user_input):
        model_s.load_state_dict(model_state_dict)
        seg_sents, css, sent_char_lens, mention_poses, quote_idxes, cut_css, name_list_index = next(
            user_input_iter)
        features, tokens_list = convert_examples_to_features(examples=css, tokenizer=tokenizer_s)

        try:
            predictions = model_s(features, sent_char_lens, mention_poses, quote_idxes, 0, "cpu",
                                tokens_list, cut_css)

            # scores, scores_false, scores_true = predictions
            scores, _, _ = predictions

            # 후처리
            scores_np = scores.detach().cpu().numpy()
            scores_list = scores_np.tolist()
            score_index = scores_list.index(max(scores_list))
            name_index = name_list_index[score_index]

            for key, val in alias2id.items():
                if val == name_index:
                    result_key = key

            # print(result_key, ins_list[i][10])
            who.append(result_key)

        except RuntimeError:
            # print('UNK', ins_list[i][10])
            who.append('UNK')

    output = making_script(text, who, ins_num)

    return output
