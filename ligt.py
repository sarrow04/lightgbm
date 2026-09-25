import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import shap
import matplotlib.pyplot as plt

st.title("LightGBM 回帰予測アプリ (SHAP対応版)")

# 1. CSVのアップロード
uploaded_file = st.file_uploader("学習用CSVデータをアップロードしてください", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("データプレビュー:", df.head())

    # 2. 目的変数の選択
    target_col = st.selectbox("予測したいターゲット変数を選択してください", df.columns)

    if st.button("モデル学習を実行"):
        with st.spinner('学習と分析を実行中です...'):
            X = df.drop(columns=[target_col])
            # カテゴリ変数を自動的に判定してcategory型に変換
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = X[col].astype('category')
                
            y = df[target_col]

            # 評価用にデータを分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'seed': 42
            }

            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10)]
            )

            # 3. 予測の実行
            preds = model.predict(X_test)
            
            # --- 精度の計算 ---
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            # 対数計算のエラーを防ぐため0以下をクリップして評価
            preds_clip = np.clip(preds, 0, None)
            y_test_clip = np.clip(y_test, 0, None)
            rmsle = np.sqrt(mean_squared_error(np.log1p(y_test_clip), np.log1p(preds_clip)))

            st.success("学習と予測が完了しました！")
            
            # 指標を並べて表示
            st.subheader("モデルの予測精度")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSLE", f"{rmsle:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAE", f"{mae:.4f}")
            col4.metric("R2 Score", f"{r2:.4f}")

            # --- 予測結果のダウンロード ---
            st.subheader("予測結果データの確認とダウンロード")
            result_df = X_test.copy()
            result_df['実際の値'] = y_test
            result_df['予測値'] = preds
            
            st.dataframe(result_df.head())

            # CSV化してダウンロードボタンを配置
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="予測結果をCSVでダウンロード",
                data=csv,
                file_name='lightgbm_predictions.csv',
                mime='text/csv',
            )

            # --- SHAP値の計算と可視化 ---
            st.subheader("SHAP値による特徴量の解釈")
            st.write("各特徴量が予測値にどのように影響を与えたかを示しています。")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # StreamlitでMatplotlibのグラフを表示
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig)
