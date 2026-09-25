import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import shap
import matplotlib.pyplot as plt
import optuna

# Optunaのログ出力を非表示にしてStreamlitのターミナルをすっきりさせる
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, X_train, y_train, X_valid, y_valid):
    # 学習データと検証データの合計件数を取得
    data_size = len(X_train) + len(X_valid)
    
    # 件数によってOptunaの探索範囲を自動分岐
    if data_size < 10000:
        # 小規模データ用（過学習を防ぐ厳しめの範囲）
        depth_range = (3, 6)
        leaves_range = (7, 31)
        min_data_range = (20, 60)
        feature_frac_range = (0.6, 0.9)
    else:
        # 大規模データ用（表現力を引き出す広い範囲）
        depth_range = (5, 12)
        leaves_range = (31, 256)
        min_data_range = (20, 100)
        feature_frac_range = (0.7, 1.0)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        
        'max_depth': trial.suggest_int('max_depth', depth_range[0], depth_range[1]),
        'num_leaves': trial.suggest_int('num_leaves', leaves_range[0], leaves_range[1]),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', min_data_range[0], min_data_range[1]),
        'feature_fraction': trial.suggest_float('feature_fraction', feature_frac_range[0], feature_frac_range[1]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    }

    # 矛盾防止：葉の数が2^max_depthを超えないように補正
    max_leaves_limit = (2 ** params['max_depth']) - 1
    if params['num_leaves'] > max_leaves_limit:
        params['num_leaves'] = max_leaves_limit

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
        num_boost_round=500
    )
    
    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse

st.title("LightGBM 自動チューニング予測アプリ")
st.write("データ件数を自動判定し、Optunaで最適なパラメータを探索します。")

# 1. CSVのアップロード
uploaded_file = st.file_uploader("学習用CSVデータをアップロードしてください", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("データプレビュー:", df.head())
    st.info(f"読み込んだデータ件数: {len(df)}行")

    # 2. 目的変数の選択
    target_col = st.selectbox("予測したいターゲット変数を選択してください", df.columns)
    
    # 探索回数の設定UI（待ち時間調整用）
    n_trials = st.slider("Optunaの探索回数 (多いほど精度が上がる可能性がありますが時間がかかります)", min_value=5, max_value=50, value=20)

    if st.button("モデル学習と最適化を実行"):
        with st.spinner('Optunaによるパラメータ最適化と学習を実行中です...'):
            X = df.drop(columns=[target_col])
            
            # カテゴリ変数を自動的に判定してcategory型に変換
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = X[col].astype('category')
                
            y = df[target_col]

            # 評価用にデータを分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- Optunaによるハイパーパラメータチューニング ---
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
            
            best_params = study.best_params
            best_params['objective'] = 'regression'
            best_params['metric'] = 'rmse'
            best_params['boosting_type'] = 'gbdt'
            best_params['verbosity'] = -1
            best_params['seed'] = 42
            
            # 矛盾補正の再適用（最終学習用）
            max_leaves_limit = (2 ** best_params['max_depth']) - 1
            if best_params['num_leaves'] > max_leaves_limit:
                best_params['num_leaves'] = max_leaves_limit

            st.success("最適パラメータの探索が完了しました！")
            with st.expander("採用された最適パラメータを確認"):
                st.json(best_params)

            # --- ベストパラメータで最終学習 ---
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            final_model = lgb.train(
                best_params,
                train_data,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                num_boost_round=1000
            )

            # --- 予測と精度の計算 ---
            preds = final_model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            # RMSLEの計算（対数計算のエラーを防ぐため0以下をクリップ）
            preds_clip = np.clip(preds, 0, None)
            y_test_clip = np.clip(y_test, 0, None)
            rmsle = np.sqrt(mean_squared_error(np.log1p(y_test_clip), np.log1p(preds_clip)))
            
            # 指標を並べて表示
            st.subheader("モデルの予測精度 (テストデータ)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSLE", f"{rmsle:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAE", f"{mae:.4f}")
            col4.metric("R2 Score", f"{r2:.4f}")

            # --- 予測結果のダウンロード ---
            st.subheader("予測結果の確認とダウンロード")
            result_df = X_test.copy()
            result_df['実際の値'] = y_test
            result_df['予測値'] = preds
            
            st.dataframe(result_df.head())

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="予測結果をCSVでダウンロード",
                data=csv,
                file_name='lightgbm_optuna_predictions.csv',
                mime='text/csv',
            )

            # --- SHAP値の計算と可視化 ---
            st.subheader("SHAP値による特徴量の解釈")
            
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_test)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig)
