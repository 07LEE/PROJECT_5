# %%
import torch
# from model.ner.ner_utils import show_tokens, ner_inference, get_ner_predictions
from module.input_process import make_name_list, make_ner_input
from module.load_model import load_ner, load_fs

fs_model = load_fs()
ner_model, checkpoint = load_ner()
checkpoint, n_args, n_tokenizer, n_optimizer, unique_tags, tag2id, id2tag = load_ner_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ner_model.to(device)
fs_model.to(device)
ner_model.eval()
fs_model.eval()

TEXT = './test/test.txt'
with open(TEXT, "r", encoding="utf-8") as f:
    text = f.read()

split_sentences = make_ner_input(text)
name_list = make_name_list(split_sentences, checkpoint)


# %%



namess = []
for i in name_list:
    namess.extend(i)

# %%
