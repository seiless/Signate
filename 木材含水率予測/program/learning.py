import joblib
import logging
import os
from sklearn.cross_decomposition import PLSRegression
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def learn(X, y, species, use_ensemble=True):
    """
    全種別を統合し、一つのグローバルモデルを学習する
    """
    logger.info(f"グローバルモデルの学習を開始します。サンプル数: {len(X)}")

    # --- モデル1: Global PLS ---
    # 全データを使用するため、成分数を少し多めに設定可能
    pls = PLSRegression(n_components=20)
    pls.fit(X, y)

    # --- モデル2: Global LightGBM ---
    lgbm = None
    if use_ensemble:
        # 未知の種別（数値）に対してもある程度対応できるよう木モデルを構築
        lgbm = LGBMRegressor(
            n_estimators=500,  # データ量が増えたため、学習回数を増加
            learning_rate=0.03,
            random_state=42,
            n_jobs=1,
            importance_type="gain",
        )
        lgbm.fit(X, y)

    trained_bundle = {
        "pls_model": pls,
        "lgbm_model": lgbm,
        "config": {"use_ensemble": use_ensemble},
    }

    joblib.dump(trained_bundle, "global_ensemble_model.pkl")
    logger.info("グローバルモデルの保存が完了しました。")

    return trained_bundle
