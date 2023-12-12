"""
NER 모델을 이용하여 작업
"""
import torch
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    model = checkpoint['model']
    tag2id = checkpoint['tag2id']
    model.to(device)
    text = text.replace(' ', '_')

    predictions, true_labels = [], []

    tokenized_sent = ner_tokenizer(text, len(text) + 2, checkpoint)
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


def ner_inference_name(tokenized_sent, pred_tags, checkpoint, name_len=5) -> list:
    """
    Name에 한해서 inference
    """
    name_list = []
    output = []
    speaker = ''
    tokenizer = checkpoint['tokenizer']

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

    for i in name_list:
        output.extend(i)

    return name_list

def make_name_list(ner_inputs, checkpoint):
    """
    문장들을 NER 돌려서 Name List 만들기.
    """
    name_list = []
    for ner_input in ner_inputs:
        tokenized_sent, pred_tags = get_ner_predictions(ner_input, checkpoint)
        names = ner_inference_name(tokenized_sent, pred_tags, checkpoint)
        name_list.append(names)

    return name_list