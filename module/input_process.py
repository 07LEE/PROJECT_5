"""
사용자 입력을 가공하는 모듈
"""
from torch.utils.data import DataLoader

# from model.find_speaker.data_prep import seg_and_mention_location, create_CSS, ISDataset
# from model.find_speaker.arguments import get_train_args
# from model.ner.ner_utils import ner_inference_name, get_ner_predictions

# def make_instance_list(text: str, ws=10) -> dict:
#     """
#     입력받은 문장을 기초적인 인스턴스 리스트로 만들어줍니다.
#     """
#     lines = text.splitlines()
#     max_line = len(lines)

#     utterance = ['"', '“']
#     instance_num = []

#     for idx, line in enumerate(lines):
#         if any(u in line for u in utterance):
#             instance_num.append(idx)

#     instance = [[] for _ in range(len(instance_num))]

#     for i, num in enumerate(instance_num):
#         if num - ws <= 0 and num + ws + 1 < max_line:
#             instance[i] += ([''] * (ws - num))
#             instance[i] +=(lines[:num + 1 + ws])
#         elif num - ws <= 0 and num + ws + 1 >= max_line:
#             instance[i] += ([''] * (ws - num))
#             instance[i] +=(lines)
#             instance[i] += ([''] * (ws * 2 - len(instance[i]) + 1))
#         elif num + ws + 1 >= max_line:
#             instance[i] +=(lines[num-ws:max_line+1])
#             instance[i] += ([''] * (num + ws + 1 - max_line))
#         else:
#             instance[i] += (lines[num-ws:num + ws + 1])

#     return instance

# def input_data_loader(instances: list, alias2id) -> DataLoader:
#     """
#     나눠진 데이터를 맞추기 위해 가공
#     """
#     data_list = []

#     for instance in instances:
#         seg_sents, candidate_mention_poses, name_list_index = seg_and_mention_location(
#             instance, alias2id)
#         css, sent_char_lens, mention_poses, quote_idxes, cut_css = create_CSS(
#             seg_sents, candidate_mention_poses, args)

#         data_list.append((seg_sents, css, sent_char_lens, mention_poses, quote_idxes,
#                           cut_css, name_list_index))

#     data_loader = DataLoader(ISDataset(data_list), batch_size=1, collate_fn=lambda x: x[0])

#     return data_loader

def make_ner_input(text, chunk_size=500) -> list:
    """
    문장을 New Lines 기준으로 나누어 줍니다.
    chunk size보다 문장이 길 경우, 마지막 문장은 뒤에서 chunk size 만큼 추가합니다.
    """
    count_text = chunk_size
    max_text = len(text)
    newline_position = []

    while count_text < max_text:
        sentence = text[:count_text]
        last_newline_position = sentence.rfind('\n')
        newline_position.append(last_newline_position)
        count_text = last_newline_position + chunk_size

    split_sentences = []
    start_num = 0

    for _, num in enumerate(newline_position):
        split_sentences.append(text[start_num:num])
        start_num = num

    if max_text % chunk_size != 0:
        f_sentence = text[max_text-500:]
        first_newline_position =  max_text-500 + f_sentence.find('\n')
        split_sentences.append(text[first_newline_position:])

    return split_sentences

def making_script(text, speaker:list, instance_num:list) -> str:
    """
    스크립트를 만드는 함수
    """
    lines = text.splitlines()
    for num, people in zip(instance_num, speaker):
        lines[num] = f'{people}: {lines[num]}'
    return lines
