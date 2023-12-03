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

from training.arguments import get_train_args
from training.bert_features import convert_examples_to_features
from training.data_prep import build_data_loader, load_data_loader
from training.load_name_list import get_alias2id
from training.train_model import CSN, KCSN
from training.training_control import adjust_learning_rate, save_checkpoint

warnings.filterwarnings(action='ignore')


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
            features = convert_examples_to_features(
                examples=CSSs, tokenizer=tokenizer)
            scores, scores_false, scores_true = model(
                features, sent_char_lens, mention_poses, quote_idxes, true_index, device)
            loss_list = [loss_fn(x.unsqueeze(0), y.unsqueeze(
                0), torch.tensor(-1.0).unsqueeze(0).to(device)) for x, y in zip(scores_false, scores_true)]

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


if __name__ == '__main__':
    # Settings
    LOG_FORMAT = "%(asctime)s [%(levelname)s]: %(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    TIMESTANP_FORMAT = '%Y%m%d-%H%M%S'

    SAVE_LOADER = 'training/data_loader'
    MODEL_NAME = 'CSN'

    # args & data path
    args = get_train_args()
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file
    name_list_path = args.name_list_path

    # checkpoint & logging
    checkpoint_dir = ''
    LOG_FATH = 'training/training_logs'

    log_dir = os.path.join(checkpoint_dir, LOG_FATH,
                           datetime.datetime.now().strftime(TIMESTANP_FORMAT))

    writer = SummaryWriter(log_dir=log_dir)
    logging_name = os.path.join(checkpoint_dir, LOG_FATH, '/training_log.log')
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                        datefmt=DATE_FORMAT, filename=logging_name)

    # device -----------------------------------------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('device : ', device)
    print('MODEL_NAME : ', MODEL_NAME)

    # alias to id ------------------------------------------------
    alias2id = get_alias2id(name_list_path)

    # %% build training, development and test data loaders ----------
    example_print = False

    if MODEL_NAME == 'CSN':
        try:
            print('load_data_loader ----------------------------------------')
            train_data = load_data_loader(f'{SAVE_LOADER}/train_data')
            dev_data = load_data_loader(f'{SAVE_LOADER}/dev_data')
            test_data = load_data_loader(f'{SAVE_LOADER}/test_data')
            example_print = True

        except FileNotFoundError:
            train_data = build_data_loader(train_file, alias2id, args, skip_only_one=True,
                                           save_filename=f'{SAVE_LOADER}/train_data', MODEL_NAME=MODEL_NAME)
            dev_data = build_data_loader(
                dev_file, alias2id, args, save_filename=f'{SAVE_LOADER}/dev_data', MODEL_NAME=MODEL_NAME)
            test_data = build_data_loader(
                test_file, alias2id, args, save_filename=f'{SAVE_LOADER}/test_data', MODEL_NAME=MODEL_NAME)
            print("---------------------------------------------------------")

        if example_print is not False:
            # example ----------------------------------------------------
            print('DEV EXAMPLE ---------------------------------------------')
            dev_test_iter = iter(dev_data)
            _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = next(
                dev_test_iter)
            print('Candidate-specific segments : ', CSSs)
            print('Nearest mention positions : ', mention_poses)

            print('TEST EXAMPLE ---------------------------------------------')
            test_test_iter = iter(test_data)
            _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = next(
                test_test_iter)
            print('Candidate-specific segments : ', CSSs)
            print('Nearest mention positions : ', mention_poses)
            print("---------------------------------------------------------")

        print("---------------------------------------------------------")

    elif MODEL_NAME == 'KCSN':
        try:
            print('load_data_loader ----------------------------------------')
            train_data = load_data_loader(f'{SAVE_LOADER}/train_Kdata')
            dev_data = load_data_loader(f'{SAVE_LOADER}/dev_Kdata')
            test_data = load_data_loader(f'{SAVE_LOADER}/test_Kdata')
            example_print = True

        except FileNotFoundError:
            train_data = build_data_loader(train_file, alias2id, args, skip_only_one=True,
                                           save_filename=f'{SAVE_LOADER}/train_Kdata', MODEL_NAME=MODEL_NAME)
            dev_data = build_data_loader(
                dev_file, alias2id, args, save_filename=f'{SAVE_LOADER}/dev_Kdata', MODEL_NAME=MODEL_NAME)
            test_data = build_data_loader(
                test_file, alias2id, args, save_filename=f'{SAVE_LOADER}/test_Kdata', MODEL_NAME=MODEL_NAME)
            print("---------------------------------------------------------")

        if example_print is not False:
            # example ----------------------------------------------------
            print('DEV EXAMPLE ---------------------------------------------')
            dev_test_iter = iter(dev_data)
            _, CSSs, sent_char_lens, mention_poses, quote_idxes, cut_css, one_hot_label, true_index, _ = next(
                dev_test_iter)
            print('Candidate-specific segments : ', CSSs)
            print('Nearest mention positions : ', mention_poses)

            print('TEST EXAMPLE ---------------------------------------------')
            test_test_iter = iter(test_data)
            _, CSSs, sent_char_lens, mention_poses, quote_idxes, cut_css, one_hot_label, true_index, _ = next(
                test_test_iter)
            print('Candidate-specific segments : ', CSSs)
            print('Nearest mention positions : ', mention_poses)
            print("---------------------------------------------------------")

    print("The number of training instances: " + str(len(train_data)))
    print("The number of development instances: " + str(len(dev_data)))
    print("The number of test instances: " + str(len(test_data)))
    print("---------------------------------------------------------")

    # initialize model ---------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)

    if MODEL_NAME == 'KCSN':
        model = KCSN(args)
    else:
        model = CSN(args)

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
    OOM_list = list()

    if MODEL_NAME == 'CSN':
        for epoch in range(start_epoch, args.num_epochs):
            acc_numerator = 0
            acc_denominator = 0
            train_loss = 0

            model.train()
            optimizer.zero_grad()

            print('Epoch: %d' % (epoch + 1))
            for i, (_, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, _) in enumerate(tqdm(train_data)):

                try:
                    features = convert_examples_to_features(
                        examples=CSSs, tokenizer=tokenizer)
                    scores, scores_false, scores_true = model(
                        features, sent_char_lens, mention_poses, quote_idxes, true_index, device)

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
                    acc_numerator += 1 if scores.max(
                        0)[1].item() == true_index else 0
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
            logging.info(
                f'Learning rate adjusted to: {adjust_learning_rate(optimizer, args.lr_decay)}')

            # Evaluation
            model.eval()

            # development stage ------------------------------------------
            overall_dev_acc, dev_avg_loss = eval(
                dev_data, 'dev', writer, epoch)

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
                overall_test_acc, test_avg_loss = eval(
                    test_data, 'test', writer, epoch)
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

    elif MODEL_NAME == 'KCSN':
        for epoch in range(start_epoch, args.num_epochs):
            acc_numerator = 0
            acc_denominator = 0
            train_loss = 0

            model.train()
            optimizer.zero_grad()

            print('Epoch: %d' % (epoch + 1))
            for i, (_, CSSs, sent_char_lens, mention_poses, quote_idxes, cut_css, one_hot_label, true_index, _) in enumerate(tqdm(train_data)):

                try:
                    features, tokens_list = convert_examples_to_features(
                        examples=CSSs, tokenizer=tokenizer, is_Kfeatures=True)
                    scores, scores_false, scores_true = model(
                        features, sent_char_lens, mention_poses, quote_idxes, true_index, device, tokens_list, cut_css)

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
                    acc_numerator += 1 if scores.max(
                        0)[1].item() == true_index else 0
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
            logging.info(
                f'Learning rate adjusted to: {adjust_learning_rate(optimizer, args.lr_decay)}')

            # Evaluation
            model.eval()

            # development stage ------------------------------------------
            overall_dev_acc, dev_avg_loss = eval(
                dev_data, 'dev', writer, epoch)

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
                overall_test_acc, test_avg_loss = eval(
                    test_data, 'test', writer, epoch)
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
