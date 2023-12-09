

```
project_root/
├-- data/
|   ├-- json/
|   └-- db_operations.py
├-- ner/
|   ├-- log/
|   ├-- test/
|   |   └-- model.pth # 테스트를 진행할 모델
|   ├-- data/
|   |   ├-- tags.pkl
|   |   └-- texts.pkl
|   └-- arguments.py
|   └-- ner_tokenize.py
|   └-- test.py
|   └-- train.py
|-- processing/
|   |-- collection.py
|   └-- preprocessing.py
├-- find_speaker/
|   ├-- train.py
|   ├-- test.py
|   ├-- test/
|   |   └-- model.pth # 테스트를 진행할 모델
|   ├-- data/
|   |   ├-- data_train
|   |   ├-- ...
|   |   ├-- training_data.txt  # 학습용 데이터 파일
|   |   └-- validation_data.txt  # 검증용 데이터 파일
|   ├-- log/
|   |   ├-- tensorborad
|   |   └-- checkpoint
|   ├-- __init__.py
|   ├-- arguments.py
|   ├-- bert_features.py
|   ├-- data_check.py
|   ├-- data_prep.py
|   ├-- load_name_list.py
|   ├-- train_model.py
|   └-- training_control.py
├-- .gitignore
├-- main.py
├-- readme.md
└-- requirements.txt
```
