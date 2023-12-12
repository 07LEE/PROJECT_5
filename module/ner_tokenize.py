"""
A:
"""
import torch


class TokenDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for tokenized data with labels.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings[idx]['input_ids']),
            'attention_mask': torch.tensor(self.encodings[idx]['attention_mask']),
            'labels': torch.tensor(self.labels[idx])
        }
        return item

    def __len__(self):
        return len(self.labels)


def ner_tokenizer(sent, max_seq_length, tokenizer):
    """
    NER 토크나이저
    """
    pad_token_id = tokenizer.pad_token_id  # 0
    cls_token_id = tokenizer.cls_token_id  # 101
    sep_token_id = tokenizer.sep_token_id  # 102

    pre_syllable = "_"
    input_ids = [pad_token_id] * (max_seq_length - 1)
    attention_mask = [0] * (max_seq_length - 1)
    token_type_ids = [0] * max_seq_length
    sent = sent[:max_seq_length-2]

    for i, syllable in enumerate(sent):
        if syllable == '_':
            pre_syllable = syllable
        # 중간 음절에는 모두 prefix를 붙입니다. (학습 데이터와 같은 형식)
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


def encode_tags(tags, max_seq_length, tag2id):
    """
    우리의 label도 truncation과 tokenizing이 필요!!
    label 역시 입력 token과 개수를 맞춰줍니다
    """
    pad_token_label_id = tag2id['O']

    tags = tags[:max_seq_length-2]
    labels = [tag2id[tag] for tag in tags]
    labels = [tag2id['O']] + labels

    padding_length = max_seq_length - len(labels)
    labels = labels + ([pad_token_label_id] * padding_length)

    return labels
