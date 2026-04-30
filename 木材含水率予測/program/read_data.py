import pandas as pd
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


def load_datasets():
    """trainデータとtestデータを読み込む関数"""
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    try:

        train_data = pd.read_csv(train_path, encoding="sjis")
        test_data = pd.read_csv(test_path, encoding="sjis")

        logger.info("データ読み込み成功")
        return train_data, test_data
    except Exception as e:
        logger.exception(e)
