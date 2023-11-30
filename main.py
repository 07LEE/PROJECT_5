# %%
import copy
import datetime
import logging
import os
import pickle
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from model.model import CSN
from utils.arguments import get_train_args
from utils.bert_features import *
from utils.data_prep import build_data_loader, load_data_loader
from utils.load_name_list import get_alias2id
from utils.training_control import *

warnings.filterwarnings(action='ignore')

# training log
LOG_FORMAT = "%(asctime)s [%(levelname)s]: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# %%
print("STARTING -----------------------------------------------")
args = get_train_args()
timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

# checkpoint & logging -----------------------------------------
checkpoint_dir = ''
log_dir = os.path.join(checkpoint_dir, 'tensorboard', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)

logging_name = os.path.join(checkpoint_dir, 'tensorboard/training_log.log')
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, filename=logging_name)

# device -----------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device : ', device)

# data files -------------------------------------------------
train_file = 'data/train_unsplit.txt'
dev_file = 'data/dev_unsplit.txt'
test_file = 'data/test_unsplit.txt'
name_list_path = 'data/name_list.txt'
print('train_file :', train_file)
print('dev_file : ', dev_file) 
print('test_file : ', test_file)
print('name_list_path : ', name_list_path)

# alias to id ------------------------------------------------
alias2id = get_alias2id(name_list_path)
print('alias2id : ', alias2id)
print("---------------------------------------------------------")

# %% build training, development and test data loaders ----------
example_print = False

try:
    print('load_data_loader ----------------------------------------')
    train_data = load_data_loader('save_data/train_data')
    dev_data = load_data_loader('save_data/dev_data')
    test_data = load_data_loader('save_data/test_data')
    example_print = True

except FileNotFoundError:
    train_data = build_data_loader(train_file, alias2id, args, skip_only_one=True, save_filename='save_data/train_data')
    dev_data = build_data_loader(dev_file, alias2id, args, save_filename='save_data/dev_data')
    test_data = build_data_loader(test_file, alias2id, args, save_filename='save_data/test_data')
    print("---------------------------------------------------------")

print("The number of training instances: " + str(len(train_data)))
print("The number of development instances: " + str(len(dev_data)))
print("The number of test instances: " + str(len(test_data)))
print("---------------------------------------------------------")

if example_print is not False:
    # example ----------------------------------------------------
    print('DEV EXAMPLE ---------------------------------------------')
    dev_test_iter = iter(dev_data)
    _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = next(dev_test_iter)
    print('Candidate-specific segments : ', CSSs)
    print('Nearest mention positions : ', mention_poses)

    print('TEST EXAMPLE ---------------------------------------------')
    test_test_iter = iter(test_data)
    _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = next(test_test_iter)
    print('Candidate-specific segments : ', CSSs)
    print('Nearest mention positions : ', mention_poses)
    print("---------------------------------------------------------")

# initialize model ---------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)
model = CSN(args)
model = model.to(device)

# initialize optimizer -----------------------------------------
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    raise ValueError("Unknown optimizer type...")

# loss criterion
loss_fn = nn.MarginRankingLoss(margin=args.margin)

# %%
def eval(eval_data, subset_name, writer, epoch):
    """
    Evaluate performance on a given subset.

    Params:
        eval_data: The set of instances to be evaluated on.
        subset_name: The name of the subset for logging.
        writer: The tensorboard writer.
        epoch: The current epoch.

    Returns:
        overall_eval_acc: Overall accuracy on the subset.
        eval_avg_loss: Average loss on the subset.
    """
    overall_eval_acc_numerator = 0
    eval_sum_loss = 0
    total_instances = len(eval_data)
    
    for _, CSSs, sent_char_lens, mention_poses, quote_idxes, _, true_index, category in eval_data:
        with torch.no_grad():
            features = convert_examples_to_features(examples=CSSs, tokenizer=tokenizer)
            scores, scores_false, scores_true = model(features, sent_char_lens, mention_poses, quote_idxes, true_index, device)
            loss_list = [loss_fn(x.unsqueeze(0), y.unsqueeze(0), torch.tensor(-1.0).unsqueeze(0).to(device)) for x, y in zip(scores_false, scores_true)]

        eval_sum_loss += sum(x.item() for x in loss_list)

        # evaluate accuracy ------------------------------------------
        correct = 1 if scores.max(0)[1].item() == true_index else 0
        overall_eval_acc_numerator += correct

    overall_eval_acc = overall_eval_acc_numerator / total_instances
    eval_avg_loss = eval_sum_loss / total_instances

    # logging ------------------------------------------
    writer.add_scalar('Loss/' + subset_name, eval_avg_loss, epoch)
    writer.add_scalar('Accuracy/' + subset_name, overall_eval_acc, epoch)
    
    logging.info(f"{subset_name}_overall_acc: {overall_eval_acc:.4f}")
    
    print(f"{subset_name}_overall_acc: {overall_eval_acc:.4f}")
    print(f"{subset_name}_overall_loss: {eval_avg_loss:.4f}")

    return overall_eval_acc, eval_avg_loss


# %% training loop -------------------------------------------------
print("Training Begins------------------------------------------")

# logging best
best_overall_dev_acc = 0
best_latent_dev_acc = 0
best_dev_loss = 0
new_best = False

# control parameters
patience_counter = 0
backward_counter = 0

# history_train_acc, history_train_loss 등 필요한 변수들을 정의
history_train_acc = list()
history_train_loss = list()

# Load checkpoint if available
start_epoch = 0
backward_counter = 0  # backward_counter 초기화 추가
checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    best_overall_dev_acc = checkpoint['best_overall_dev_acc']
    best_dev_loss = checkpoint['best_dev_loss']
    backward_counter = checkpoint['backward_counter']  # 저장된 backward_counter 로드
    print(f"Resuming training from epoch {start_epoch}")

# start epoch
OOM_list = list()
for epoch in range(start_epoch, args.num_epochs):
    acc_numerator = 0
    acc_denominator = 0
    train_loss = 0

    model.train()
    optimizer.zero_grad()
    
    print('Epoch: %d' % (epoch + 1))
    for i, (_, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, _) in enumerate(tqdm(train_data)):
        
        try:
            features = convert_examples_to_features(examples=CSSs, tokenizer=tokenizer)
            scores, scores_false, scores_true = model(features, sent_char_lens, mention_poses, quote_idxes, true_index, device)

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
            OOM_list.append([epoch+1, i])

    acc = acc_numerator / acc_denominator
    train_loss /= len(train_data)

    # logging ------------------------------------------
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', acc, epoch)

    logging.info(f'train_acc: {acc:.4f}')
    print(f'train_acc: {acc:.4f}')
    print(f'train_loss: {train_loss:.4f}')
    print("---------------------------------------------------------")
    
    history_train_acc.append(acc)
    history_train_loss.append(train_loss)

    # adjust learning rate after each epoch
    logging.info(f'Learning rate adjusted to: {adjust_learning_rate(optimizer, args.lr_decay)}')

    # Evaluation
    model.eval()

    # development stage ------------------------------------------
    overall_dev_acc, dev_avg_loss = eval(dev_data, 'dev', writer, epoch)

    # save the model with best performance
    if overall_dev_acc > best_overall_dev_acc:
        best_overall_dev_acc = overall_dev_acc
        best_dev_loss = dev_avg_loss
        patience_counter = 0
        new_best = True
    else:
        patience_counter += 1
        new_best = False

    # only save the model which outperforms the former best on development set
    if new_best:
        # test stage
        overall_test_acc, test_avg_loss = eval(test_data, 'test', writer, epoch)
        try:
            info_json = {"epoch": epoch}
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_overall_dev_acc': best_overall_dev_acc,
                'best_dev_loss': best_dev_loss,
                'backward_counter': backward_counter  # backward_counter 저장
            }, dirname=checkpoint_dir, info_json=info_json)
        except Exception as e:
            print(e)

    # early stopping
    if patience_counter > args.patience:
        print("Early stopping...")
        break
    print('------------------------------------------------------')

print('best_overall_dev_acc :', best_overall_dev_acc)
print('overall_test_acc : ', overall_test_acc)

# 학습 루프 완전히 종료 후에 모델 저장
save_path = os.path.join(checkpoint_dir, 'final_model.pth')
torch.save(model.state_dict(), save_path)
print(f"Final model saved to {save_path}")

# 텐서보드 로그 폴더 닫기
writer.close()


# %%

# # %%
# if __name__ == '__main__':
#     # run several times and calculate average accuracy and standard deviation
#     dev = []
#     test = []
#     for i in range(3):    
#         dev_acc, test_acc = train()
#         dev.append(dev_acc)
#         test.append(test_acc)

#     dev = np.array(dev)
#     test = np.array(test)

#     dev_mean = np.mean(dev)
#     dev_std = np.std(dev)
#     test_mean = np.mean(test)
#     test_std = np.std(test)

#     print(str(dev_mean) + '(±' + str(dev_std) + ')')
#     print(str(test_mean) + '(±' + str(test_std) + ')')


# %% 저장된 모델 상태 사전 로드 -------------------------------------------
model_state_dict = torch.load("final_model.pth")

# 새로운 모델 인스턴스 생성 및 상태 사전 로드
model = CSN(args)
model.load_state_dict(model_state_dict)

# 입력 데이터 준비
check = 'data/check.txt'
check_data = build_data_loader(check, alias2id, args)
check_data_iter = iter(check_data)

_, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, _ = next(check_data_iter)

# 예측 수행
predictions = model(features=features, sent_char_lens=sent_char_lens, mention_poses=mention_poses, quote_idxes=quote_idxes, true_index=true_index, device='cpu')
scores, scores_false, scores_true = predictions

# 예측 출력
print('predictions : ', predictions)
print('scores : ', scores)
print('scores_false : ', scores_false)
print('scores_true : ', scores_true)

# %%
