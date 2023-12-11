"""
Author: 
"""
# %%
import datetime
import logging
import os
import warnings

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from .arguments import get_train_args
from .bert_features import convert_examples_to_features
from .data_prep import build_data_loader, load_data_loader
from .load_name_list import get_alias2id
from .train_model import CSN, KCSN
from .training_control import adjust_learning_rate, save_checkpoint

warnings.filterwarnings(action='ignore')

def run():
    """Just Run"""
    # Settings
    log_format = "%(asctime)s [%(levelname)s]: %(message)s"
    data_format = '%Y-%m-%d %H:%M:%S'
    timestamp_format = '%Y%m%d-%H%M%S'
    save_loader = 'training/data'

    # args & data path
    args = get_train_args()
    model_name = args.model_name
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file
    name_list_path = args.name_list_path

    # checkpoint & logging
    checkpoint_dir = args.checkpoint_dir
    log_fath = args.training_logs

    log_dir = os.path.join(log_fath, datetime.datetime.now().strftime(timestamp_format))

    writer = SummaryWriter(log_dir=log_dir)
    logging_name = os.path.join(log_fath, '/training_log.log')
    logging.basicConfig(level=logging.INFO, format=log_format,
                        datefmt=data_format, filename=logging_name)

    # DEVICE & alias to id ---------------------------------------
    print("---------------------------------------------------------")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    alias2id = get_alias2id(name_list_path)

    print('DEVICE : ', device)
    print('MODEL_NAME : ', model_name)

    # build data loaders ------------------------------------------
    try:
        print('load data loader ----------------------------------------')
        train_data = load_data_loader(f'{save_loader}/data_train')
        dev_data = load_data_loader(f'{save_loader}/data_dev')
        test_data = load_data_loader(f'{save_loader}/data_test')
    except FileNotFoundError:
        print('build data loader ----------------------------------------')
        train_data = build_data_loader(train_file, alias2id, args,
                                       save_name=f'{save_loader}/data_train',
                                       skip_only_one=True)
        dev_data = build_data_loader(dev_file, alias2id, args,
                                     save_name=f'{save_loader}/data_dev',)
        test_data = build_data_loader(test_file, alias2id, args,
                                      save_name=f'{save_loader}/data_test')

    print("---------------------------------------------------------")
    print('DEV EXAMPLE : ')
    dev_test_iter = iter(dev_data)
    _, css, _, mention_poses, _, _, _, _, _, name_list_index = next(
        dev_test_iter)
    print('- Candidate-specific segments : ', css)
    print('- Nearest mention positions : ', mention_poses)
    print('- Name list index : ', name_list_index)

    print('TEST EXAMPLE : ')
    test_test_iter = iter(test_data)
    _, css, _, mention_poses, _, _, _, _, _, name_list_index = next(
        test_test_iter)
    print('- Candidate-specific segments : ', css)
    print('- Nearest mention positions : ', mention_poses)
    print('- Name list index : ', name_list_index)

    print("---------------------------------------------------------")
    print("The number of training instances: " + str(len(train_data)))
    print("The number of development instances: " + str(len(dev_data)))
    print("The number of test instances: " + str(len(test_data)))
    print("---------------------------------------------------------")

    # initialize model ---------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)

    if model_name == 'KCSN':
        model = KCSN(args)
    elif model_name == 'CSN':
        model = CSN(args)
    else:
        raise ValueError("Unknown model type...")
    model = model.to(device)

    # initialize optimizer -----------------------------------------
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer type...")

    # loss criterion -----------------------------------------------
    loss_fn = nn.MarginRankingLoss(margin=args.margin)

    # training loop -------------------------------------------------
    print("Training Begins------------------------------------------")

    # logging best
    best_overall_dev_acc = 0
    best_dev_loss = 0
    best_test_acc = 0
    best_test_loss = 0

    # control parameters
    patience_counter = 0
    backward_counter = 0

    # history_train_acc, history_train_loss 등 필요한 변수들을 정의
    history_train_acc = []
    history_train_loss = []
    history_test_acc = []
    history_test_loss = []
    oom_list = []

    # Load checkpoint if available -------------------------------
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
        # 저장된 backward_counter 로드
        backward_counter = checkpoint['backward_counter']
        print(f"Resuming training from epoch {start_epoch}")

    # start epoch -------------------------------------------------
    for epoch in range(start_epoch, args.num_epochs):
        acc_numerator = 0
        acc_denominator = 0
        train_loss = 0

        model.train()
        optimizer.zero_grad()

        print(f'Epoch: {epoch + 1}')
        for i, (_, css, sent_char_lens, mention_poses, quote_idxes, cut_css, _,
                true_index, _, _) in enumerate(tqdm(train_data)):
            try:
                features, tokens_list = convert_examples_to_features(
                    examples=css, tokenizer=tokenizer)
                scores, scores_false, scores_true = model(
                    features, sent_char_lens, mention_poses, quote_idxes,
                    true_index, device, tokens_list, cut_css)

                # backward propagation and weights update
                for x, y in zip(scores_false, scores_true):
                    # compute loss
                    loss = loss_fn(x.unsqueeze(0), y.unsqueeze(
                        0), torch.tensor(-1.0).unsqueeze(0).to(device))
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

            except (RuntimeError, TypeError) as e:
                oom_list.append([epoch+1, i, f'{e}', 'train'])

        save_path = os.path.join(checkpoint_dir, f'{epoch}_model.pth')
        torch.save(model.state_dict(), save_path)

        acc = acc_numerator / acc_denominator
        train_loss /= len(train_data)

        # logging ------------------------------------------
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)

        logging.info('train_acc: %.4f', acc)
        print(f'Train Acc: {acc:.4f}')
        print(f'Train Loss: {train_loss:.4f}')
        print("---------------------------------------------------------")

        history_train_acc.append(acc)
        history_train_loss.append(train_loss)

        # adjust learning rate after each epoch
        logging.info('Learning rate adjusted to: %s',
                     adjust_learning_rate(optimizer, args.lr_decay))

        # Evaluation
        model.eval()

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

            for _, css, scl, mp, quote_idxes, cc, _, true_index, _, _ in eval_data:
                try:
                    with torch.no_grad():
                        features, tokens_list = convert_examples_to_features(
                            examples=css, tokenizer=tokenizer)
                        scores, scores_false, scores_true = model(features, scl, mp, quote_idxes,
                                                                true_index, device, tokens_list, cc)
                        loss_list = [loss_fn(x.unsqueeze(0), y.unsqueeze(0),torch.tensor(-1.0
                            ).unsqueeze(0).to(device)) for x, y in zip(scores_false, scores_true)]
                    eval_sum_loss += sum(x.item() for x in loss_list)

                    # evaluate accuracy ------------------------------------------
                    correct = 1 if scores.max(0)[1].item() == true_index else 0
                    overall_eval_acc_numerator += correct

                except (RuntimeError, TypeError) as es:
                    oom_list.append([epoch+1, i, f'{es}', f'{subset_name}'])

            overall_eval_acc = overall_eval_acc_numerator / total_instances
            eval_avg_loss = eval_sum_loss / total_instances

            # logging ------------------------------------------
            writer.add_scalar('Loss/' + subset_name, eval_avg_loss, epoch)
            writer.add_scalar('Accuracy/' + subset_name, overall_eval_acc, epoch)
            logging.info('%s_overall_acc: %.4f', subset_name, overall_eval_acc)
            print(f"{subset_name}_overall_acc: {overall_eval_acc:.4f}")
            print(f"{subset_name}_overall_loss: {eval_avg_loss:.4f}")

            return overall_eval_acc, eval_avg_loss

        # development stage ------------------------------------------------------------------------
        overall_dev_acc, dev_avg_loss = eval(dev_data, 'dev', writer, epoch)

        # save the model with best performance
        if overall_dev_acc > best_overall_dev_acc:
            best_overall_dev_acc = overall_dev_acc
            best_dev_loss = dev_avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # test stage -------------------------------------------------------------------------------
        test_acc, test_loss = eval(test_data, 'test', writer, epoch)
        history_test_acc.append(test_acc)
        history_test_loss.append(test_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_loss = test_loss

        # save checkpint
        try:
            info_json = {"epoch": epoch}
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_overall_dev_acc': best_overall_dev_acc,
                'best_dev_loss': best_dev_loss,
                'backward_counter': backward_counter}, dirname=checkpoint_dir, info_json=info_json)
        except Exception as e:
            print(e)

        # early stopping
        if patience_counter > args.patience:
            print("Early stopping...")
            break
        print('------------------------------------------------------')

    print(f'best_overall_dev_acc: {best_overall_dev_acc:.4f}')
    print(f'Best Test Acc: {best_test_acc:.4f}, Best Test Loss: {best_test_loss:.4f}')

    # 학습 루프 완전히 종료 후에 모델 저장
    save_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")

    # Error List 저장
    texts = os.path.join(log_fath, 'oom_list.txt')
    with open(texts, 'w', encoding='utf-8') as f:
        for oom in oom_list:
            f.write(f"{oom}\n")

    # 텐서보드 로그 폴더 닫기
    writer.close()

if __name__ == '__main__':
    run()

# %%
