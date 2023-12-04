# Generate BERT features.
import re


class InputFeatures(object):
    """
    Inputs of the BERT model.
    """
    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, tokenizer, is_Kfeatures=False):
    """
    Convert textual segments into word IDs.

    params
        examples: the raw textual segments in a list.
        tokenizer: a BERT Tokenizer object.
        is_Kfeatures: a flag to indicate whether to return tokens_list or not.

    return
        features: BERT features in a list.
        tokens_list: a list of tokens (only when is_Kfeatures is True).
    """
    features = []
    tokens_list = []

    for (ex_index, example) in enumerate(examples):
        if is_Kfeatures:
            tokens = tokenizer.tokenize(example)
            tokens_list.append(tokens)
        else:
            tokens = list()
            for ex in example:
                tokens += [letter for letter in re.sub('\s', '', ex)]

        new_tokens = []
        input_type_ids = []

        new_tokens.append("[CLS]")
        input_type_ids.append(0)
        new_tokens += tokens
        input_type_ids += [0] * len(tokens)
        new_tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        input_mask = [1] * len(input_ids)

        features.append(
            InputFeatures(
                tokens=new_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))

    if is_Kfeatures:
        return features, tokens_list
    else:
        return features