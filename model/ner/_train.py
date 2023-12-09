"""
A
"""
# %%
import datetime
import pickle
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, BertForTokenClassification
from sklearn.model_selection import train_test_split

from arguments import get_train_args
from ner_tokenize import ner_tokenizer, encode_tags, TokenDataset

if __name__ == '__main__':
    # Args --------------------------------------------------
    args = get_train_args()

    MODEL_NAME = args.bert_pretrained_dir
    batch_size = args.per_device_train_batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs

    TIME_FORMAT = '%Y%m%d-%H%M%S'
    time_now = datetime.datetime.now().strftime(TIME_FORMAT)

    # DEVICE ------------------------------------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('DEVICE: ', device)

    # Load --------------------------------------------------
    with open('data/texts.pkl', 'rb') as f:
        texts = pickle.load(f)
    with open('data/tags.pkl', 'rb') as f:
        tags = pickle.load(f)

    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    # Train Test Split
    train_texts, test_texts, train_tags, test_tags = train_test_split(
        texts, tags, test_size=.2)

    # Tokenizer --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pad_token_label_id = tag2id['O']    # tag2id['O']
    cls_token_label_id = tag2id['O']
    sep_token_label_id = tag2id['O']

    # Tokenizing --------------------------------------------------
    tokenized_train_sentences = []
    tokenized_test_sentences = []
    for text in train_texts:    # 전체 데이터를 tokenizing 합니다.
        tokenized_train_sentences.append(ner_tokenizer(text, 128, tokenizer))
    for text in test_texts:
        tokenized_test_sentences.append(ner_tokenizer(text, 128, tokenizer))

    # Tag Encoding -----------------------------------------------------
    train_labels = []
    test_labels = []
    for tag in train_tags:
        train_labels.append(encode_tags(tag, 128, tag2id))
    for tag in test_tags:
        test_labels.append(encode_tags(tag, 128, tag2id))

    # Dataset 준비 -------------------------------------------------------
    train_dataset = TokenDataset(tokenized_train_sentences, train_labels)
    test_dataset = TokenDataset(tokenized_test_sentences, test_labels)

    # Model --------------------------------------------------------------
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(unique_tags))
    model.to(device)

    # Optimizer 및 DataLoader 설정
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    train_dataloader = DataLoader(train_dataset,
                                batch_size=args.per_device_train_batch_size,
                                shuffle=True)

    # 훈련 루프
    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if step % args.logging_steps == 0:
                print(f"Epoch {epoch}, Step {step}: Loss = {loss.item()}")

    # %% 모델 저장
    model.to('cpu')
    PATH = f'{time_now}.pth'
    torch.save({'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'tokenizer': tokenizer,
                'unique_tags': unique_tags,
                'tag2id': tag2id,
                'id2tag': id2tag}, PATH)

# %%
