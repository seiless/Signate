import joblib
import logging

logger = logging.getLogger(__name__)


def predict(X_test, species_test):
    """
    統合されたグローバルモデルを使用して予測を行う
    """
    try:
        bundle = joblib.load("global_ensemble_model.pkl")
    except FileNotFoundError:
        logger.error("モデルファイルが見つかりません。")
        raise

    pls = bundle["pls_model"]
    lgbm = bundle["lgbm_model"]
    use_ensemble = bundle["config"]["use_ensemble"]

    # グローバルモデルなので、全データを一度に投入可能（コンペ規則が許す場合）
    # もし「1サンプルずつ」の厳格な制限がある場合はループで処理

    # PLS予測
    preds_pls = pls.predict(X_test).flatten()

    if use_ensemble and lgbm is not None:
        # LGBM予測
        preds_lgbm = lgbm.predict(X_test)
        # 最終アンサンブル (PLS 0.6 : LGBM 0.4 など調整)
        final_preds = (preds_pls * 0.6) + (preds_lgbm * 0.4)
    else:
        final_preds = preds_pls

    return final_preds
