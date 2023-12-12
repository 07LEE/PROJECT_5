"""
A
"""
from pathlib import Path
import re
import torch
import numpy as np

from .ner_tokenize import ner_tokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_ner_predictions(text, checkpoint, device=DEVICE):
    """
    Predicts named entity tags for the given text using the provided NER model and tokenizer.
    """
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']
    tag2id = checkpoint['tag2id']

    model.eval()
    text = text.replace(' ', '_')

    predictions, true_labels = [], []

    tokenized_sent = ner_tokenizer(text, len(text) + 2, tokenizer)
    input_ids = torch.tensor(tokenized_sent['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(tokenized_sent['attention_mask']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(tokenized_sent['token_type_ids']).unsqueeze(0).to(device)

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

def show_tokens(text, checkpoint):
    """
    문장을 넣어서 토큰을 확인합니다.
    """
    tokenizer = checkpoint['tokenizer']
    tokenized_sent, pred_tags = get_ner_predictions(text, checkpoint)

    print(pred_tags)
    print('TOKEN \t TAG')
    print("===========")

    for i, tag in enumerate(pred_tags):
        # if tag != 'O':
        print("{:^5}\t{:^5}".format(
            tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'][i]), tag))

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

def extract_entities(text, model, device, tokenizer):
    entities = []
    current_entity = ""
    current_tag = ""

    for i, tag in enumerate(pred_tags):
        token = tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'][i]).replace("##", "")

        if tag.startswith("B-"):
            if current_entity and current_tag != "O":
                entities.append(current_entity)
            current_entity = token
            current_tag = tag[2:]
        elif tag.startswith("I-"):
            if current_entity and tag[2:] == current_tag:
                current_entity += token
        else:
            if current_entity and current_tag != "O":
                entities.append(current_entity)
            current_entity = ""
            current_tag = ""

    if current_entity and current_tag != "O":
        entities.append(current_entity)

    return entities

def ner_inference(tokenized_sent, pred_tags, checkpoint) -> (list, dict):
    tokenizer = checkpoint['tokenizer']
    names = []
    scene = {'장소': [], '시간': []}
    target = ''
    c_tag = None
    for i, tag in enumerate(pred_tags):
        token = tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'][i]).replace('#', '')
        if tag != 'O':
            if tag.startswith('B'):
                if c_tag:
                    if pred_tags[i-1].startswith('I'+c_tag):
                        target += token.replace('_', ' ')
                else:
                    if c_tag == 'PER': #이전까지의 태그를 확인해서 업데이트한 후 초기화해줍니다.
                        names.append(target)
                    elif c_tag == 'TIM' or c_tag == 'DAT':
                        scene['시간'].append(target)
                    elif c_tag =='LOC':
                        scene['장소'].append(target)
                    c_tag = tag[2:]
                    target = token
            else:
                target += token.replace('_', ' ')

    return list(set(names)), scene

def ner_inference_name(tokenized_sent, pred_tags, checkpoint, name_len=5) -> list:
    """
    Name에 한해서 inference.
    """
    tokenizer = checkpoint['tokenizer']
    name_list = []
    speaker = ''

    for i, tag in enumerate(pred_tags):
        token = tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'][i]).replace('#', '')
        if 'PER' in tag:
            if 'B' in tag and speaker != '':
                name_list.append(speaker)
                speaker = ''
            speaker += token

        elif speaker != '' and tag != pred_tags[i-1]:
            tmp = speaker
            found_name = False
            for j in range(name_len):
                if i + j < len(tokenized_sent['input_ids']):
                    token = tokenizer.convert_ids_to_tokens(
                        tokenized_sent['input_ids'][i+j]).replace('#', '')
                    tmp += token
                    if tmp in name_list:
                        name_list.append(tmp)
                        found_name = True
                        break

            if not found_name:
                name_list.append(speaker)
            speaker = ''

    return name_list
