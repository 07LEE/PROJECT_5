from module.load_model import load_fs, load_ner
from module.input_process import make_ner_input
from module.ner_utils import make_name_list

ner_model, checkpoint = load_ner()

with open('test/test.txt', "r", encoding="utf-8") as f:
    file_content = f.read()


content = make_ner_input(file_content)
name_list = make_name_list(content, checkpoint)