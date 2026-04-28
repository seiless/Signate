import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


def load_datasets():
    """train.csv와 test.csv를 읽어 반환하는 함수"""
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_data = pd.read_csv(train_path, encoding="sjis")
    test_data = pd.read_csv(test_path, encoding="sjis")

    print("데이터 로드 완료!")
    return train_data, test_data
