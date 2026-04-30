import joblib
import numpy as np
import pandas as pd
import logging

# ロギングの設定
logger = logging.getLogger(__name__)


def predict(X_test, species_test):
    """
    評価用データに対して学習済みモデルを用いて予測を行う関数

    引数:
        X_test: 前処理済みの評価用データ
        species_test: 評価用データの種別情報

    戻り値:
        predictions: 最終的な含水率の予測値（numpy配列）
    """
    try:
        # 保存されたモデル一式をロード
        bundle = joblib.load("ensemble_model_bundle.pkl")
    except FileNotFoundError:
        logger.error(
            "学習済みモデルファイルが見つかりません。先に learn を実行してください。"
        )
        raise

    models_dict = bundle["models"]
    use_ensemble = bundle["config"]["use_ensemble"]

    final_preds = []

    # コンペ規則に基づき、各スペクトルを独立して処理
    for i in range(len(X_test)):
        sp = species_test.iloc[i]
        # 1サンプルを抽出して形状を整える (1, n_features)
        single_x = X_test[i].reshape(1, -1)

        if sp in models_dict:
            # 当該種別専用のモデルを取得
            target_models = models_dict[sp]

            # PLSによる予測
            p_pls = target_models["pls"].predict(single_x)[0][0]

            if use_ensemble and "lgbm" in target_models:
                # LightGBMによる予測
                p_lgbm = target_models["lgbm"].predict(single_x)[0]

                # アンサンブル: PLSとLGBMの結果を重み付け平均 (比率は検証結果に基づき調整)
                # 一般的に分光データはPLSが強いため、PLSに高い重みを置く
                combined = (p_pls * 0.7) + (p_lgbm * 0.3)
                final_preds.append(combined)
            else:
                final_preds.append(p_pls)
        else:
            # 学習データに存在しない種別がテストデータに現れた場合のフォールバック処理
            # 全モデルの平均値を使用するなどの対応
            logger.warning(
                f"未知の種別を検出しました (Species {sp})。デフォルト予測を適用します。"
            )
            all_pls_preds = [
                m["pls"].predict(single_x)[0][0] for m in models_dict.values()
            ]
            final_preds.append(np.mean(all_pls_preds))

    return np.array(final_preds)
