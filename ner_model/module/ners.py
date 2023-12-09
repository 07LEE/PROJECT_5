"""
A
"""
from pathlib import Path
import re
import torch
import numpy as np

def show_tokens(text, model, device, tokenizer, tag2id):
    """
    문장을 넣어서 토큰을 확인합니다.
    """
    text = text.replace(' ', '_')
    predictions , true_labels = [], []

    tokenized_sent = ner_tokenizer(text, len(text)+2, tokenizer)
    input_ids = torch.tensor(tokenized_sent['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(tokenized_sent['attention_mask']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(tokenized_sent['token_type_ids']).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

    logits = outputs['logits']
    logits = logits.detach().cpu().numpy()
    label_ids = token_type_ids.cpu().numpy()

    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.append(label_ids)

    pred_tags = [list(tag2id.keys())[p_i] for p in predictions for p_i in p]

    print('TOKEN \t TAG')
    print("===========")

    for i, tag in enumerate(pred_tags):
        # if tag != 'O':
        print("{:^5}\t{:^5}".format(
            tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'][i]), tag))

def ner_tokenizer(sent, max_seq_length, tokenizer):
    """
    기존 토크나이저는 wordPiece tokenizer로 tokenizing 결과를 반환.
    데이터 처럼 tokenizer도 음절단위로 맞춤. (unk 토큰 줄이기 위함)
    """
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
            # 중간 음절에는 모두 prefix를 붙입니다. (학습 데이터와 같은 형식)
            syllable = '##' + syllable
        pre_syllable = syllable

        input_ids[i] = (tokenizer.convert_tokens_to_ids(syllable))
        attention_mask[i] = 1

    input_ids = [cls_token_id] + input_ids
    input_ids[len(sent)+1] = sep_token_id
    attention_mask = [1] + attention_mask
    attention_mask[len(sent)+1] = 1

    return {"input_ids":input_ids,
            "attention_mask":attention_mask,
            "token_type_ids":token_type_ids}

def read_file(file_list):
    """
    ner train을 위한 코드
    """
    token_docs = []
    tag_docs = []
    for file_path in file_list:
        # print("read file from ", file_path)
        file_path = Path(file_path)
        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                if line[0:1] == "$" or line[0:1] == ";" or line[0:2] == "##":
                    continue
                try:
                    token = line.split('\t')[0]
                    tag = line.split('\t')[3]   # 2: pos, 3: ner
                    for i, syllable in enumerate(token):    # 음절 단위로 잘라서
                        tokens.append(syllable)
                        modi_tag = tag
                        if i > 0:
                            if tag[0] == 'B':
                                modi_tag = 'I' + tag[1:]    # BIO tag를 부착할게요 :-)
                        tags.append(modi_tag)
                except:
                    print(line)
            token_docs.append(tokens)
            tag_docs.append(tags)

    return token_docs, tag_docs

def encode_tags(tags, max_seq_length, tag2id, pad_token_label_id):
    """
    우리의 label도 truncation과 tokenizing이 필요!!
    label 역시 입력 token과 개수를 맞춰줍니다
    """
    tags = tags[:max_seq_length-2]
    labels = [tag2id[tag] for tag in tags]
    labels = [tag2id['O']] + labels

    padding_length = max_seq_length - len(labels)
    labels = labels + ([pad_token_label_id] * padding_length)
    print(labels)
    return labels
