"""
NER 모델을 이용하여 작업
"""
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


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
        first_newline_position = max_text-500 + f_sentence.find('\n')
        split_sentences.append(text[first_newline_position:])

    return split_sentences


def ner_tokenizer(sent, max_seq_length, checkpoint):
    """
    NER 토크나이저
    """
    tokenizer = checkpoint['tokenizer']

    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    pre_syllable = "_"
    input_ids = [pad_token_id] * (max_seq_length - 1)
    attention_mask = [0] * (max_seq_length - 1)
    token_type_ids = [0] * max_seq_length
    sent = sent[:max_seq_length-2]

    for i, syllable in enumerate(sent):
        if syllable == '_':
            pre_syllable = syllable
        if pre_syllable != "_":
            syllable = '##' + syllable
        pre_syllable = syllable

        input_ids[i] = tokenizer.convert_tokens_to_ids(syllable)
        attention_mask[i] = 1

    input_ids = [cls_token_id] + input_ids[:-1] + [sep_token_id]
    attention_mask = [1] + attention_mask[:-1] + [1]

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids}


def get_ner_predictions(text, checkpoint):
    """
    tokenized_sent, pred_tags 만들기
    """
    import numpy as np

    model = checkpoint['model']
    tag2id = checkpoint['tag2id']
    model.to(device)
    text = text.replace(' ', '_')

    predictions, true_labels = [], []

    tokenized_sent = ner_tokenizer(text, len(text) + 2, checkpoint)
    input_ids = torch.tensor(
        tokenized_sent['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(
        tokenized_sent['attention_mask']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(
        tokenized_sent['token_type_ids']).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

    logits = outputs['logits']
    logits = logits.detach().cpu().numpy()
    label_ids = token_type_ids.cpu().numpy()

    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.append(label_ids)

    pred_tags = [list(tag2id.keys())[p_i] for p in predictions for p_i in p]

    return tokenized_sent, pred_tags


def ner_inference_name(tokenized_sent, pred_tags, checkpoint, name_len=5) -> list:
    """
    Name에 한해서 inference
    """
    name_list = []
    speaker = ''
    tokenizer = checkpoint['tokenizer']

    for i, tag in enumerate(pred_tags):
        token = tokenizer.convert_ids_to_tokens(
            tokenized_sent['input_ids'][i]).replace('#', '')
        if 'PER' in tag:
            if 'B' in tag and speaker != '':
                name_list.append(speaker)
                speaker = ''
            speaker += token

        elif speaker != '' and tag != pred_tags[i-1]:
            if speaker in name_list:
                name_list.append(speaker)
            else:
                tmp = speaker
                found_name = False
                print(f'{speaker}에 의문이 생겨 확인해봅니다.')
                for j in range(name_len):
                    if i + j < len(tokenized_sent['input_ids']):
                        token = tokenizer.convert_ids_to_tokens(
                            tokenized_sent['input_ids'][i+j]).replace('#', '')
                        tmp += token
                        print(f'{speaker} 뒤로 나온 {j} 번째 까지 확인한결과, {tmp} 입니다')
                        if tmp in name_list:
                            name_list.append(tmp)
                            found_name = True
                            print(f'명단에 {tmp} 가 존재하여, {speaker} 대신 추가하였습니다.')
                            break

                if not found_name:
                    name_list.append(speaker)
                    print(f'찾지 못하여 {speaker} 를 추가하였습니다.')
                speaker = ''

    return name_list


def make_name_list(ner_inputs, checkpoint):
    """
    문장들을 NER 돌려서 Name List 만들기.
    """
    name_list = []
    for ner_input in ner_inputs:
        tokenized_sent, pred_tags = get_ner_predictions(ner_input, checkpoint)
        names = ner_inference_name(tokenized_sent, pred_tags, checkpoint)
        name_list.extend(names)

    return name_list


def show_name_list(name_list):
    """
    사용자 친화적으로 보여주기용
    """
    from collections import Counter
    name = Counter(name_list).most_common()

    return name
