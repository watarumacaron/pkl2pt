# pkl2pt

## pkl2ptとは☕️
これは、pklの重みを抽出し、pytorchのモデルにフィットするように変換するコードである。styleganのdisciminatorの重みを変換したかったため、discriminatorを具体例として使用している。他のモデルを試したい場合は、pythonファイルの編集が必要であるが、今後の参考になればと思い作成した。

## 使い方
### 1. 重みの抽出
まず、pklモデルから重みを抽出する必要があるため、extract_param.pyを実行する。
実行環境は、pklモデルの重みをロードすることができる（pklモデルが動く）環境を使用する。
引数は、pklモデルのpath（.pkl）と保存する際の名前（.pkl）である。
```
python extract_param.py pkl_model_path output_name
```

### 2. .ptに変換
次に、1.で抽出した重みをpytorchモデルにフィットするように変換する。
実行環境は、pytochモデルがう動く環境を使用する。
引数は、抽出した重みのpath（.pkl）と保存する際のモデルの名前（.pt）である。
```
python convert_param.py pkl_params_path output_model_name
```
