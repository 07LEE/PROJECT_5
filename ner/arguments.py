from argparse import ArgumentParser

# Your variables
ROOT_DIR = ""
BERT_PRETRAINED_DIR = "bert-base-multilingual-cased"
CHECKPOINT_DIR = 'training/checkpoint'
DATA_PREFIX = "training/data"
LOG_FATH = 'training/training_logs'

def get_train_args():
    """
    return
        args: training arguments
    """
    parser = ArgumentParser(description='I_S', allow_abbrev=False)

    # BERT
    parser.add_argument('--bert_pretrained_dir', type=str, default=BERT_PRETRAINED_DIR)

    # Train
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=int, default=3e-5)

    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64)

    # Logging
    parser.add_argument('--output_dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--logging_dir', type=str, default=LOG_FATH)
    parser.add_argument('--logging_steps', type=int, default=100)

    parser.add_argument('--save_total_limit', type=int, default=5)

    args, _ = parser.parse_known_args()

    return args
