import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import shap
import matplotlib.pyplot as plt

st.title("LightGBM 手動チューニング予測アプリ")
st.write("サイドバーの数値を調整して、LightGBMの挙動と精度をリアルタイムに確認できます。")

# --- サイドバー：手動パラメータ設定 ---
st.sidebar.header("🔧 パラメータ手動調整")
st.sidebar.write("データ件数に合わせて調整してください。")

p_max_depth = st.sidebar.slider("max_depth (木の深さ)", min_value=1, max_value=15, value=5, help="浅い(3~5)と過学習を防ぎ、深いと表現力が上がります。")
p_num_leaves = st.sidebar.slider("num_leaves (葉の最大数)", min_value=2, max_value=128, value=31, help="大きくすると精度が上がりますが過学習しやすくなります。")
p_min_data = st.sidebar.slider("min_data_in_leaf (葉の最小データ数)", min_value=1, max_value=100, value=20, help="データが少ない場合は大きめに設定してブレーキをかけます。")
p_lr = st.sidebar.number_input("learning_rate (学習率)", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
p_feature_frac = st.sidebar.slider("feature_fraction (特徴量サンプリング割合)", min_value=0.4, max_value=1.0, value=0.8, step=0.1)

# 矛盾防止：葉の数が2^max_depthを超えないように画面上で補正
max_leaves_limit = (2 ** p_max_depth) - 1
if p_num_leaves > max_leaves_limit:
    st.sidebar.warning(f"⚠️ max_depth={p_max_depth} の理論限界に合わせて、num_leavesを {max_leaves_limit} に自動補正します。")
    p_num_leaves = max_leaves_limit

# 1. CSVのアップロード
uploaded_file = st.file_uploader("学習用CSVデータをアップロードしてください", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("データプレビュー:", df.head())
    st.info(f"読み込んだデータ件数: {len(df)}行 / 特徴量: {len(df.columns)-1}列")

    # 2. 目的変数の選択
    target_col = st.selectbox("予測したいターゲット変数を選択してください", df.columns)

    if st.button("設定したパラメータで学習を実行"):
        with st.spinner('学習を実行中です...'):
            X = df.drop(columns=[target_col])
            
            # カテゴリ変数を自動的に判定してcategory型に変換
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = X[col].astype('category')
                
            y = df[target_col]

            # 評価用にデータを分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- モデル学習 ---
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'seed': 42,
                'max_depth': p_max_depth,
                'num_leaves': p_num_leaves,
                'min_data_in_leaf': p_min_data,
                'learning_rate': p_lr,
                'feature_fraction': p_feature_frac
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                num_boost_round=1000
            )

            # --- 予測と精度の計算 ---
            preds = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            preds_clip = np.clip(preds, 0, None)
            y_test_clip = np.clip(y_test, 0, None)
            rmsle = np.sqrt(mean_squared_error(np.log1p(y_test_clip), np.log1p(preds_clip)))
            
            st.success(f"学習完了！ (ストップしたラウンド: {model.best_iteration})")
            
            # 指標を並べて表示
            st.subheader("モデルの予測精度")
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
            
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="予測結果をCSVでダウンロード",
                data=csv,
                file_name='lightgbm_manual_predictions.csv',
                mime='text/csv',
            )

            # --- SHAP値の計算と可視化 ---
            st.subheader("SHAP値による特徴量の解釈")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig)
