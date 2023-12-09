"""
Module: bert_features_generator

This module provides functions to generate BERT features from textual segments.
"""
import re
from arguments import get_train_args

args = get_train_args()

class InputFeatures:
    """
    Represents the inputs of the BERT model.

    Attributes:
        tokens (list): List of tokens.
        input_ids (list): List of input IDs.
        input_mask (list): List of input masks.
        input_type_ids (list): List of input type IDs.
    """
    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

    def get_tokens(self):
        """
        Get the list of tokens.

        Returns:
            list: List of tokens.
        """
        return self.tokens

    def get_input_ids(self):
        """
        Get the list of input IDs.

        Returns:
            list: List of input IDs.
        """
        return self.input_ids

    def get_input_mask(self):
        """
        Get the list of input masks.

        Returns:
            list: List of input masks.
        """
        return self.input_mask

    def get_input_type_ids(self):
        """
        Get the list of input type IDs.

        Returns:
            list: List of input type IDs.
        """
        return self.input_type_ids

def convert_examples_to_features(examples, tokenizer):
    """
    Convert textual segments into word IDs.

    params:
        examples: the raw textual segments in a list.
        tokenizer: a BERT Tokenizer object.

    return:
        features: BERT features in a list.
        tokens_list: a list of tokens (only when is_kfeatures is True).
    """
    features = []
    tokens_list = []

    for (ex_index, example) in enumerate(examples):
        if args.model_name == 'KCSN':
            tokens = tokenizer.tokenize(example)
            tokens_list.append(tokens)
        else:
            tokens = []
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

    return features, tokens_list
