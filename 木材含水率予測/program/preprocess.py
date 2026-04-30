import pandas as pd
import numpy as np
import os
import logging
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

# ロギングの設定
logger = logging.getLogger(__name__)


class SpectralPreprocessor:
    def __init__(self, window_length=15, polyorder=2, deriv=2):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.margin = (window_length - 1) // 2
        self.scaler = StandardScaler()
        self.spec_cols = None

    def _apply_sg_and_trim(self, data):
        sg_data = savgol_filter(
            data, self.window_length, self.polyorder, self.deriv, axis=1
        )
        return sg_data[:, self.margin : -self.margin]

    def process_train_data(self, df):
        self.spec_cols = [c for c in df.columns if c[0].isdigit()]
        X_raw = df[self.spec_cols].values
        X_sg = self._apply_sg_and_trim(X_raw)
        X_scaled = self.scaler.fit_transform(X_sg)

        # [Global] 分光データに 'species number' を1つの変数として結合
        # Testに未知の種別が出るため、One-Hotではなく数値のまま、あるいは埋め込み用として保持
        X_combined = np.column_stack([X_scaled, df["species number"].values])

        return X_combined, df["含水率"], df["species number"]

    def process_test_data(self, df):
        if self.spec_cols is None:
            raise ValueError("Train data で fit してください。")

        X_raw = df[self.spec_cols].values
        X_sg = self._apply_sg_and_trim(X_raw)
        X_scaled = self.scaler.transform(X_sg)

        # [Global] Testデータも同様に種別番号を結合
        X_combined = np.column_stack([X_scaled, df["species number"].values])

        return X_combined, df["species number"], df.get("sample number")


def load_datasets():
    """
    program フォルダから見て ../data/ フォルダにある CSV を読み込む
    """
    # 現在のファイルの場所 (program/) からの相対パスを設定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    try:
        # 日本語を含むデータのため 'sjis' を指定
        train_data = pd.read_csv(train_path, encoding="sjis")
        test_data = pd.read_csv(test_path, encoding="sjis")
        logger.info(f"データの読み込みに成功しました: {data_dir}")
        return train_data, test_data
    except Exception as e:
        logger.exception(f"データ読み込みエラー: {e}")
        raise
