# %%
import tqdm as notebook_tqdm
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from model.model import CSN
from utils.load_name_list import *
from utils.data_prep import build_data_loader
from utils.arguments import get_train_args
from utils.bert_features import *


warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    print("RST--------------------------------------------------")
    args = get_train_args()

    # device -----------------------------------------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('Device : ', device)
    print("-----------------------------------------------------")

    # data files -------------------------------------------------
    train_file = 'data/train_unsplit.txt'
    dev_file = 'data/dev_unsplit.txt'
    test_file = 'data/test_unsplit.txt'
    name_list_path = 'data/name_list.txt'
    print('train file : ', train_file)
    print('dev_file : ', dev_file)
    print('test_file : ', test_file)
    print('name_list_path : ', name_list_path)
    print("-----------------------------------------------------")

    # alias to id ------------------------------------------------
    alias2id = get_alias2id(name_list_path)
    print('alias2id : ', alias2id)
    print("-----------------------------------------------------")

    # initialize model --------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    model = CSN(args)
    model = model.to(device)
    
    # initialize model
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-5)
    loss_fn = nn.MarginRankingLoss(margin=1.0)

    # Data Lines check --------------------------------------------
    # okt = Okt()
    # for alias in alias2id:
    #     okt.nouns(alias)

    with open(train_file, 'r', encoding='utf-8') as fin:
        data_lines = fin.readlines()
        
    # # Make user Dict ----------------------------------------------    
    # user_dict_list = list()
    # for alias in alias2id:
    #     user_dict_list.append(alias)

    # with open('user_dict.txt', 'w', encoding='utf-8') as f:
    #     for word in user_dict_list:
    #         f.write(word + '\tNNG\n')

    print('Data Lines : ', len(data_lines))
    print("-----------------------------------------------------")

    # build_data_loader --------------------------------------------
    train_data = build_data_loader(train_file, alias2id, args, skip_only_one=True)
    print("The number of training instances: " + str(len(train_data)))
    print("-----------------------------------------------------")

    dev_data = build_data_loader(dev_file, alias2id, args)
    print("The number of development instances: " + str(len(dev_data)))
    print("-----------------------------------------------------")

    test_data = build_data_loader(test_file, alias2id, args)
    print("The number of test instances: " + str(len(test_data)))
    print("-----------------------------------------------------")

    print('---------------------DEV EXAMPLE---------------------')
    dev_test_iter = iter(dev_data)
    _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = dev_test_iter.next()
    print('Candidate-specific segments:', CSSs)
    print('Nearest mention positions:', mention_poses)
    print('Quote Idxes:', quote_idxes)
    print('One Hot Label:', one_hot_label)
    print('True Index:', true_index)
    print('Category:', category)

    test_test_iter = iter(test_data)
    print('---------------------TEST EXAMPLE--------------------')
    _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = test_test_iter.next()
    print('Candidate-specific segments:', CSSs)
    print('Nearest mention positions:', mention_poses)
    print('Quote Idxes:', quote_idxes)
    print('One Hot Label:', one_hot_label)
    print('True Index:', true_index)
    print('Category:', category)

    print('##############DEV EXAMPLE#################')
    dev_test_iter = iter(dev_data)
    _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = dev_test_iter.next()

    print('Candidate-specific segments:')
    print(CSSs)
    print("-----------------------------------------------------")

    print('Nearest mention positions:')
    print(mention_poses)
    print("-----------------------------------------------------")

    test_test_iter = iter(test_data)
    print('##############TEST EXAMPLE#################')
    _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = test_test_iter.next()
    print('Candidate-specific segments:')
    print(CSSs)
    print('Nearest mention positions:')
    print(mention_poses)
    print("-----------------------------------------------------")

# %%

# print('tokenizer : ', tokenizer)
# print("-----------------------------------------------------")
# print("optimizer : ", optimizer)
# print("-----------------------------------------------------")
# print('Loss Fn : ', loss_fn)
# print("-----------------------------------------------------")


# %%

if __name__ == '__main__':

    best_overall_dev_acc = 0
    best_explicit_dev_acc = 0
    best_implicit_dev_acc = 0
    best_latent_dev_acc = 0
    best_dev_loss = 0
    new_best = False

    # control parameters
    patience_counter = 0
    backward_counter = 0

    acc_numerator = 0
    acc_denominator = 0
    train_loss = 0

    model.train()
    optimizer.zero_grad()

#%%

for i, (_, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, _) in enumerate(train_data):
    
    # print('CSSs : ', CSSs)
    # print('sent_char_lens : ', sent_char_lens)
    # print('mention_poses : ', mention_poses)
    # print('quote_idxes : ', quote_idxes)
    # print('one_hot_label : ', one_hot_label)
    # print('true_index : ', true_index)    
    # print("-----------------------------------------------------")

    try:
        features = convert_examples_to_features(examples=CSSs, tokenizer=tokenizer)
        scores, scores_false, scores_true = model(features, sent_char_lens, mention_poses, quote_idxes, true_index, device)
        print('scores :', scores)

        # backward propagation and weights update
        for x, y in zip(scores_false, scores_true):
            # compute loss
            loss = loss_fn(x.unsqueeze(0), y.unsqueeze(0), torch.tensor(-1.0).unsqueeze(0).to(device))
            train_loss += loss.item()
            
            # backward propagation
            loss /= args.batch_size
            loss.backward(retain_graph=True)
            backward_counter += 1

            # update parameters
            if backward_counter % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        # training accuracy
        acc_numerator += 1 if scores.max(0)[1].item() == true_index else 0
        acc_denominator += 1

    except RuntimeError:
        print('OOM occurs...')

acc = acc_numerator / acc_denominator
train_loss /= len(train_data)
# %%

