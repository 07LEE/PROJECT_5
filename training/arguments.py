# Pre-set arguments
from argparse import ArgumentParser
import sys


def get_train_args():
    """
    return
        args: training arguments
    """
    parser = ArgumentParser(description='I_S', allow_abbrev=False)

    # Your variables
    ROOT_DIR = ""
    BERT_PRETRAINED_DIR = "klue/roberta-large"
    CHECKPOINT_DIR = ""
    DATA_PREFIX = "./data"

    # source ${ROOT_DIR}/.bashrc

    # Parsing arguments
    parser.add_argument('--model_name', type=str, default='CSN')

    # Model
    parser.add_argument('--pooling_type', type=str, default='max_pooling')
    parser.add_argument('--classifier_intermediate_dim', type=int, default=100)
    parser.add_argument('--nonlinear_type', type=str, default='tanh')

    # BERT
    parser.add_argument('--bert_pretrained_dir', type=str, default=BERT_PRETRAINED_DIR)

    # Training
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=2e-5)  # Change to your desired value
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)  # Change to your desired value
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10)

    # Data for training, development, and test
    parser.add_argument('--train_file', type=str, default=f'{DATA_PREFIX}train_unsplit.txt')
    parser.add_argument('--dev_file', type=str, default=f'{DATA_PREFIX}dev_unsplit.txt')
    parser.add_argument('--test_file', type=str, default=f'{DATA_PREFIX}test_unsplit.txt')
    parser.add_argument('--name_list_path', type=str, default=f'{DATA_PREFIX}/name_list.txt')
    parser.add_argument('--ws', type=int, default=10)

    parser.add_argument('--length_limit', type=int, default=510)

    # save checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)

    args, unknown = parser.parse_known_args()

    return args
