import pandas as pd
import os
import logging
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

# ロギングの設定
logger = logging.getLogger(__name__)


class SpectralPreprocessor:
    def __init__(self, window_length=15, polyorder=2, deriv=2):
        """
        引数:
            window_length: Savitzky-Golay フィルタのウィンドウ幅 (奇数である必要あり)
            polyorder: 多項式の次数
            deriv: 微分の次数 (含水率予測には2次微分が推奨される)
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.margin = (window_length - 1) // 2
        self.scaler = StandardScaler()
        self.spec_cols = None  # 学習時に確定した分光データのカラムリストを保持

    def _apply_sg_and_trim(self, data):
        """
        Savitzky-Golay 2次微分を適用し、境界部分の歪みを除去する内部関数
        """
        # axis=1 は波長方向（カラム方向）にフィルタを適用することを意味する
        sg_data = savgol_filter(
            data, self.window_length, self.polyorder, self.deriv, axis=1
        )

        # 境界部分（margin）をカットして、ノイズの影響を排除する
        return sg_data[:, self.margin : -self.margin]

    def process_train_data(self, df):
        """
        学習用データの読み込み、前処理基準の学習(fit)、および変換(transform)を行う
        """
        # 分光データのカラムを特定（数字で始まるカラムを対象とする）
        self.spec_cols = [c for c in df.columns if c[0].isdigit()]

        X_raw = df[self.spec_cols].values
        X_sg = self._apply_sg_and_trim(X_raw)

        # 学習データの平均と標準偏差を保存（StandardScaler の学習）
        X_scaled = self.scaler.fit_transform(X_sg)

        # 目的変数（含水率）と種別情報を返す
        return X_scaled, df["含水率"], df["species number"]

    def process_test_data(self, df):
        """
        評価用データに対し、学習時の基準（Scaler, Margin）を用いて変換のみを行う
        コンペ規則「未知の1スペクトルのみから予測可能か」に適合する設計
        """
        if self.spec_cols is None:
            raise ValueError(
                "まず学習用データで process_train_data を実行し、基準を確定させてください。"
            )

        X_raw = df[self.spec_cols].values
        X_sg = self._apply_sg_and_trim(X_raw)

        # 学習時と同じ基準でスケーリングを適用
        X_scaled = self.scaler.transform(X_sg)

        # 予測に必要な種別情報とサンプル番号を返す
        return X_scaled, df["species number"], df.get("sample number")


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
