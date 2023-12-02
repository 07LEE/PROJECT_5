

```
project_root/
|-- app/
|   |-- data_management/
|   |   |-- __init__.py
|   |   |-- models.py
|   |   └-- db_operations.py
|   └-- utils/
|       └-- common_utils.py
|-- data/
|   └-- json/
|-- training/
|   └-- data_loader/  # 데이터 로딩 관련 파일 저장
|   |   |-- file1
|   |   |-- file2
|   |   └-- ...
|   └-- datasets/
|   |   |-- training_data.txt  # 학습용 데이터 파일
|   |   └-- validation_data.txt  # 검증용 데이터 파일
|   └-- outputs/
|   |   └--model_checkpoints
|   └-- training_log/
|   |   └--tensorborad
|   ├-- arguments.py
|   ├-- bert_features.py
|   ├-- data_prep.py
|   ├--load_name_list.py
|   └--training_control.py
├-- .gitignore
├-- main.py
├-- train.py
├-- readme.md
└-- requirements.txt
```


