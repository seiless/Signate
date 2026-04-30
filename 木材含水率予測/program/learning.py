import joblib
import os
import numpy as np
import logging
from sklearn.cross_decomposition import PLSRegression
from lightgbm import LGBMRegressor

# ロギングの設定
logger = logging.getLogger(__name__)

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def learn(X, y, species, use_ensemble=True):
    """
    種別（species）ごとに個別のモデル（PLSおよびLGBM）を学習する関数

    引数:
        X: 前処理済みの説明変数（分光データ）
        y: 目的変数（含水率）
        species: 各サンプルの種別情報
        use_ensemble: LGBMを含めたアンサンブルを行うかどうかのフラグ

    戻り値:
        trained_bundle: 学習済みモデルと設定を含む辞書
    """
    # モデルと設定を保持する辞書
    trained_bundle = {
        "models": {},
        "config": {"use_ensemble": use_ensemble, "features_count": X.shape[1]},
    }

    unique_species = species.unique()
    logger.info(f"学習を開始します。対象種別数: {len(unique_species)}")

    for sp in unique_species:
        # 当該種別のデータのみを抽出
        idx = species == sp
        X_sp, y_sp = X[idx], y[idx]

        sp_models = {}

        # --- モデル1: PLS (分光データの線形的な化学的特性を捉える) ---
        # サンプル数に応じて成分数を調整（最大10）
        n_comp = min(10, len(y_sp) - 1)
        if n_comp < 1:
            n_comp = 1

        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_sp, y_sp)
        sp_models["pls"] = pls

        # --- モデル2: LightGBM (非線形なパターンや複雑な残差を捉える) ---
        if use_ensemble:
            # 比較的小規模なデータセットに適したパラメータ設定
            lgbm = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                importance_type="gain",
                random_state=42,
                verbose=-1,
            )
            lgbm.fit(X_sp, y_sp)
            sp_models["lgbm"] = lgbm

        trained_bundle["models"][sp] = sp_models
        logger.info(f"Species {sp}: 学習完了 (サンプル数: {len(y_sp)})")

    # 学習済みモデル群をファイルとして保存
    model_path = "ensemble_model_bundle.pkl"
    joblib.dump(trained_bundle, model_path)
    logger.info(f"全モデルの保存が完了しました: {model_path}")

    return trained_bundle
