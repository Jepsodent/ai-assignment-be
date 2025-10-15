Dataset: https://drive.google.com/drive/folders/13LrHmjdZbnCTp9reK9cIDQDfUPhcN81p?usp=sharing

Files & Folders
/data -> our dataset, /data/train is for training model and /data/val is for validating model
/train.py -> create and train the model according to our dataset
/model/densenet121_tb.h5 -> our trained model that has been created from train.py
/utils.py -> utility function, this contains the get_prediction function
/main.py -> main funciton, this contians API logic

Commands:
pip install -r requirement.txt -> to install requirements
python train.py -> train and create model
uvicorn main:app --reload -> run main function

How to run program:
1. If model doesn't exist, run python train.py
2. uvicorn main:app --reload -> create server, visit /docs for swagger documentation
