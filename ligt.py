import streamlit as st
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

st.set_page_config(page_title="LightGBM Parquet App", layout="wide")
st.title("LightGBM モデル学習 & 診断アプリ")

# 1. データ読み込み
st.sidebar.header("1. データの読み込み")
# 修正点: iPhoneでグレーアウトしないよう、type制限を外しました
uploaded_file = st.sidebar.file_uploader("Parquetファイルをアップロード")

if uploaded_file is not None:
    try:
        # Parquetの読み込み
        df = pd.read_parquet(uploaded_file)
        st.subheader("データプレビュー")
        st.dataframe(df.head(10))
        
        # 2. 設定
        st.sidebar.header("2. タスクと特徴量の設定")
        target_col = st.sidebar.selectbox("ターゲット変数", df.columns)
        task_type = st.sidebar.radio("タスクの種類", ["回帰", "二値分類", "多クラス分類"])
        
        features = [c for c in df.columns if c != target_col]
        X = df[features]
        y = df[target_col]
        
        # カテゴリ変数の簡易エンコーディング
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category')
            
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. ハイパーパラメータ調整
        st.sidebar.header("3. ハイパーパラメータ")
        learning_rate = st.sidebar.slider("learning_rate", 0.001, 0.3, 0.05, 0.005)
        num_leaves = st.sidebar.slider("num_leaves", 7, 127, 31, 2)
        max_depth = st.sidebar.slider("max_depth", -1, 20, -1, 1)
        n_estimators = st.sidebar.slider("n_estimators (学習回数)", 10, 1000, 100, 10)
        
        if st.button("モデルを学習する"):
            # LightGBM用のパラメータ設定
            params = {
                'learning_rate': learning_rate,
                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'verbose': -1
            }
            
            if task_type == "回帰":
                params['objective'] = 'regression'
                params['metric'] = 'rmse'
            elif task_type == "二値分類":
                params['objective'] = 'binary'
                params['metric'] = 'binary_logloss'
            else:
                params['objective'] = 'multiclass'
                params['num_class'] = len(y.unique())
                params['metric'] = 'multi_logloss'

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
            
            evals_result = {}
            
            with st.spinner("学習中..."):
                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=n_estimators,
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=['train', 'valid'],
                    callbacks=[lgb.record_evaluation(evals_result)]
                )
                
            st.success("学習が完了しました！")
            
            # 4. 学習曲線の描画
            st.subheader("学習曲線 (Learning Curve)")
            metric_name = params['metric']
            train_loss = evals_result['train'][metric_name]
            valid_loss = evals_result['valid'][metric_name]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train_loss, label='Train Loss')
            ax.plot(valid_loss, label='Validation Loss')
            ax.set_xlabel("Iterations")
            ax.set_ylabel(metric_name)
            ax.set_title(f"Learning Curve ({metric_name})")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            # 5. 適正診断アドバイス
            st.subheader("モデルの適正診断アドバイス")
            
            final_train_loss = train_loss[-1]
            final_valid_loss = valid_loss[-1]
            min_valid_loss = min(valid_loss)
            
            # 簡単なヒューリスティックによる診断
            if final_valid_loss > final_train_loss * 1.5 and min_valid_loss < final_valid_loss:
                st.error("⚠️ 過学習 (**Overfitting**) の可能性が高いです。")
                st.write("Validationの誤差が途中で上昇に転じているか、Trainとの乖離が大きすぎます。")
                st.markdown("- 「**learning_rate**」 を下げる\n- 「**num_leaves**」 や 「**max_depth**」 を小さくしてモデルをシンプルにする\n- **n_estimators** を Validation Loss が最小になった付近で止める (**Early Stopping** の目安)")
                
            elif final_train_loss > np.mean(train_loss[:int(n_estimators*0.1)]) * 0.9:
                st.warning("⚠️ 未学習 (**Underfitting**) の可能性が高いです。")
                st.write("Trainの誤差が十分に下がっていません。モデルがデータのパターンを学習できていない状態です。")
                st.markdown("- 「**learning_rate**」 を少し上げる\n- 「**num_leaves**」 や 「**max_depth**」 を大きくして表現力を上げる\n- **n_estimators** の上限を増やす")
                
            else:
                st.success("✅ 学習は概ね適正に進行しています。")
                st.write("TrainとValidationの誤差がともに減少しており、極端な乖離も見られません。")
                st.markdown("- さらに精度を上げる場合は、新しい特徴量の追加(**Feature Engineering**)などを検討してください。")
                
    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました。Parquet形式のファイルか確認してください。\n\n詳細なエラー: {e}")

else:
    st.info("サイドバーからParquet形式のデータをアップロードしてください。")
