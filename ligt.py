import streamlit as st
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import numpy as np

st.set_page_config(page_title="LightGBM Parquet App", layout="wide")
st.title("LightGBM モデル学習 & 診断アプリ")

# 1. データ読み込み
st.sidebar.header("1. データの読み込み")
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
        task_type = st.sidebar.radio("タスクの種類", ["回帰 (売上予測など)", "二値分類", "多クラス分類"])
        
        features = [c for c in df.columns if c != target_col]
        X = df[features]
        y = df[target_col]
        
        # カテゴリ変数の簡易エンコーディング（Pandasの警告対策で 'str' を追加）
        for col in X.select_dtypes(include=['object', 'category', 'str']).columns:
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
                'verbose': -1,
                'random_state': 42
            }
            
            if "回帰" in task_type:
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
            
            # タブで結果を分ける
            tab1, tab2, tab3 = st.tabs(["評価指標と予測結果", "学習曲線と診断", "予測データのダウンロード"])

            # テストデータに対する予測の実行
            y_pred_prob = model.predict(X_test)
            
            if "回帰" in task_type:
                y_pred = y_pred_prob
            elif task_type == "二値分類":
                y_pred = (y_pred_prob > 0.5).astype(int)
            else: # 多クラス分類
                y_pred = np.argmax(y_pred_prob, axis=1)

            # --- Tab 1: 評価指標と予測結果 ---
            with tab1:
                st.subheader("テストデータでの評価指標")
                
                if "回帰" in task_type:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # RMSLEの計算 (売上がマイナスになるのを防ぐため0にクリップして計算)
                    y_test_safe = np.clip(y_test, 0, None)
                    y_pred_safe = np.clip(y_pred, 0, None)
                    rmsle = np.sqrt(mean_squared_error(np.log1p(y_test_safe), np.log1p(y_pred_safe)))
                    
                    # RMSLEを一番左に目立たせて配置
                    col1.metric("RMSLE (Kaggle指標)", f"{rmsle:.4f}")
                    col2.metric("RMSE", f"{rmse:.4f}")
                    col3.metric("MAE", f"{mae:.4f}")
                    col4.metric("R2 Score", f"{r2:.4f}")
                    
                    # アプリ内での指標の見方解説
                    with st.expander("💡 指標（スコア）の見方・目安を開く"):
                        st.markdown("""
                        - **RMSLE (Root Mean Squared Logarithmic Error)**:  
                          **Kaggle「Store Sales」の公式指標です。** `0` に近いほど優秀です。  
                          誤差の「絶対的な大きさ」ではなく「比率」を見ます。また、実際の売上より少なく予測（過小評価/品切れリスク）してしまった場合に、より重いペナルティを与える特徴があります。
                        - **RMSE (二乗平均平方根誤差)**:  
                          `0` に近いほど優秀です。売上個数など、元のデータと同じ単位でのズレの大きさを表します。大きく外した予測があると一気に数値が悪化します。
                        - **MAE (平均絶対誤差)**:  
                          `0` に近いほど優秀です。純粋な「ズレの平均値」なので、直感的に分かりやすい指標です。
                        - **R2 Score (決定係数)**:  
                          `1.0` に近いほど優秀です（最大1.0）。モデルがデータをどれくらい正確に説明できているかを表す割合です。0.5を下回ると精度が低く、0.7以上ならある程度良いモデルとされます。
                        """)
                    
                    st.subheader("実測値 vs 予測値")
                    fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
                    ax_pred.scatter(y_test, y_pred, alpha=0.5)
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    ax_pred.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                    # 文字化け対策のため、日本語を削除して英語のみに修正
                    ax_pred.set_xlabel("True Values")
                    ax_pred.set_ylabel("Predictions")
                    ax_pred.set_title("True vs Predicted Values")
                    st.pyplot(fig_pred)

                else: # 分類
                    col1, col2, col3, col4 = st.columns(4)
                    avg_method = 'binary' if task_type == "二値分類" else 'macro'
                    
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                    rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                    f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                    
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col2.metric("Precision", f"{prec:.4f}")
                    col3.metric("Recall", f"{rec:.4f}")
                    col4.metric("F1 Score", f"{f1:.4f}")
                    
                    with st.expander("💡 指標（スコア）の見方・目安を開く"):
                        st.markdown("""
                        - **Accuracy (正解率)**: 全データの中で、正しく予測できた割合です。`1.0` に近いほど優秀です。
                        - **Precision (適合率)**: 「正」と予測した中で、本当に「正」だった割合。「誤報を減らしたい」時に重視します。
                        - **Recall (再現率)**: 実際の「正」の中で、正しく予測できた割合。「見逃しを防ぎたい」時に重視します。
                        - **F1 Score**: PrecisionとRecallのバランスを取った指標。どちらも高いモデルを作りたい時の総合評価に使います。
                        """)
                    
                    st.subheader("混同行列 (Confusion Matrix)")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_xlabel('Predicted Label')
                    ax_cm.set_ylabel('True Label')
                    ax_cm.set_title('Confusion Matrix')
                    st.pyplot(fig_cm)

            # --- Tab 2: 学習曲線と診断 ---
            with tab2:
                st.subheader("学習曲線 (Learning Curve)")
                metric_name = params['metric']
                train_loss = evals_result['train'][metric_name]
                valid_loss = evals_result['valid'][metric_name]
                
                fig_lc, ax_lc = plt.subplots(figsize=(10, 5))
                ax_lc.plot(train_loss, label='Train Loss')
                ax_lc.plot(valid_loss, label='Validation Loss')
                ax_lc.set_xlabel("Iterations")
                ax_lc.set_ylabel(metric_name)
                ax_lc.set_title(f"Learning Curve ({metric_name})")
                ax_lc.legend()
                ax_lc.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig_lc)
                
                # 適正診断アドバイス
                st.subheader("モデルの適正診断アドバイス")
                
                final_train_loss = train_loss[-1]
                final_valid_loss = valid_loss[-1]
                min_valid_loss = min(valid_loss)
                
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

            # --- Tab 3: 予測データのダウンロード ---
            with tab3:
                st.subheader("テストデータの予測結果")
                
                result_df = X_test.copy()
                result_df['True_Label'] = y_test
                result_df['Prediction'] = y_pred
                if "回帰" not in task_type:
                     if task_type == "二値分類":
                         result_df['Prediction_Prob'] = y_pred_prob
                     else:
                         result_df['Max_Prob'] = np.max(y_pred_prob, axis=1)

                st.dataframe(result_df.head(20))
                
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="予測結果をCSVでダウンロード",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )

    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました。Parquet形式のファイルか確認してください。\n\n詳細なエラー: {e}")

else:
    st.info("サイドバーからParquet形式のデータをアップロードしてください。")
