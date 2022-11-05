## Dataset
- Unzip data.zip to `./data`
    ```sh
    unzip data.zip -d ./data
    ```
- Folder structure
    ```
    .
    ├── data
    │   ├── test/
    │   └── unlabeled/
    ├── eval.py
    ├── finetune.py
    ├── model.py
    ├── pretrain.py
    ├── readfile.py
    ├── Readme.md
    ├── requirements.txt
    └── score_function.py
    ```

## Environment
- Python 3.6.13 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python pretrain.py
python finetune.py
```

## Make Prediction
```sh
python eval.py
```
The prediction file is `result.npy`.


## Schedule
```sh
python schedule.py
```
You can use this file to make training and testing together
