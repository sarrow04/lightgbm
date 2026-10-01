import streamlit as st
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # 【追加】SHAPライブラリ
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
        
        # 逆対数変換のオプション
        apply_expm1 = False
        if "回帰" in task_type:
            apply_expm1 = st.sidebar.checkbox(
                "予測結果を逆対数変換 (expm1) して元のスケールに戻す", 
                value=False, 
                help="事前にターゲット変数を対数変換(log1p)している場合、チェックを入れるとグラフやダウンロードデータを元の売上個数などに戻して出力します。"
            )

        features = [c for c in df.columns if c != target_col]
        X = df[features]
        y = df[target_col]
        
        # 日付（DateTime）型の列を自動的に除外する処理
        datetime_cols = X.select_dtypes(include=['datetime', 'datetimetz', 'datetime64']).columns
        if len(datetime_cols) > 0:
            st.warning(f"⚠️ 日付型の列 ({', '.join(datetime_cols)}) が検出されました。LightGBMは日付を直接扱えないため、自動的に特徴量から除外しました。")
            X = X.drop(columns=datetime_cols)
        
        # カテゴリ変数の簡易エンコーディング
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
            
            # 【変更】タブを4つに増やす
            tab1, tab2, tab3, tab4 = st.tabs(["評価指標と予測結果", "学習曲線と診断", "予測データのDL", "SHAP値 (特徴量の解釈)"])

            # テストデータに対する予測の実行
            y_pred_prob = model.predict(X_test)
            
            y_test_eval = y_test.copy()
            
            if "回帰" in task_type:
                y_pred = y_pred_prob
                if apply_expm1:
                    y_pred = np.expm1(y_pred)
                    y_test_eval = np.expm1(y_test_eval)
            elif task_type == "二値分類":
                y_pred = (y_pred_prob > 0.5).astype(int)
            else:
                y_pred = np.argmax(y_pred_prob, axis=1)

            # --- Tab 1: 評価指標と予測結果 ---
            with tab1:
                st.subheader("テストデータでの評価指標")
                
                if "回帰" in task_type:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred))
                    mae = mean_absolute_error(y_test_eval, y_pred)
                    r2 = r2_score(y_test_eval, y_pred)
                    
                    y_test_safe = np.clip(y_test_eval, 0, None)
                    y_pred_safe = np.clip(y_pred, 0, None)
                    rmsle = np.sqrt(mean_squared_error(np.log1p(y_test_safe), np.log1p(y_pred_safe)))
                    
                    col1.metric("RMSLE (Kaggle指標)", f"{rmsle:.4f}")
                    col2.metric("RMSE", f"{rmse:.4f}")
                    col3.metric("MAE", f"{mae:.4f}")
                    col4.metric("R2 Score", f"{r2:.4f}")
                    
                    with st.expander("💡 指標（スコア）の見方・目安を開く"):
                        st.markdown("""
                        - **RMSLE**: Kaggle「Store Sales」公式指標。`0` に近いほど優秀。
                        - **RMSE**: `0` に近いほど優秀。大きく外した予測があると悪化しやすい。
                        - **MAE**: `0` に近いほど優秀。純粋なズレの平均値。
                        - **R2 Score**: `1.0` に近いほど優秀。モデルの当てはまりの良さ。
                        """)
                    
                    st.subheader("実測値 vs 予測値")
                    fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
                    ax_pred.scatter(y_test_eval, y_pred, alpha=0.5)
                    min_val = min(y_test_eval.min(), y_pred.min())
                    max_val = max(y_test_eval.max(), y_pred.max())
                    ax_pred.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                    ax_pred.set_xlabel("True Values")
                    ax_pred.set_ylabel("Predictions")
                    ax_pred.set_title("True vs Predicted Values")
                    st.pyplot(fig_pred)

                else: # 分類
                    col1, col2, col3, col4 = st.columns(4)
                    avg_method = 'binary' if task_type == "二値分類" else 'macro'
                    
                    acc = accuracy_score(y_test_eval, y_pred)
                    prec = precision_score(y_test_eval, y_pred, average=avg_method, zero_division=0)
                    rec = recall_score(y_test_eval, y_pred, average=avg_method, zero_division=0)
                    f1 = f1_score(y_test_eval, y_pred, average=avg_method, zero_division=0)
                    
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col2.metric("Precision", f"{prec:.4f}")
                    col3.metric("Recall", f"{rec:.4f}")
                    col4.metric("F1 Score", f"{f1:.4f}")
                    
                    st.subheader("混同行列 (Confusion Matrix)")
                    cm = confusion_matrix(y_test_eval, y_pred)
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

            # --- Tab 3: 予測データのダウンロード ---
            with tab3:
                st.subheader("テストデータの予測結果")
                result_df = X_test.copy()
                result_df['True_Label'] = y_test_eval 
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
            
            # --- Tab 4: SHAP値 (特徴量の解釈) ---
            with tab4:
                st.subheader("SHAP値による特徴量の重要度")
                st.markdown("AIが予測を行う際に、**どのデータ（列）がどれくらい予測結果に影響を与えたか**を視覚的に示します。")
                
                with st.spinner("SHAP値を計算・描画中... (データが多いと少し時間がかかります)"):
                    try:
                        # SHAPの計算オブジェクトを作成
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test)
                        
                        st.write("#### 📊 Summary Plot (サマリープロット)")
                        st.markdown("""
                        - **上にある特徴量ほど**、予測に対して重要な（影響力が大きい）ことを示します。
                        - **点の色**：赤は「その値が高い」、青は「その値が低い」ことを意味します。
                        - **横軸の位置**：中心(0)より右側（プラス方向）なら予測値を押し上げ、左側（マイナス方向）なら予測値を押し下げたことを意味します。
                        """)
                        
                        plt.figure(figsize=(10, 6))
                        # タスクによってshap_valuesの形が異なる（リストで返る場合がある）ための処理
                        if isinstance(shap_values, list):
                            # 二値分類の場合などはクラス1（正例）の方のSHAP値をプロット
                            shap.summary_plot(shap_values[1] if len(shap_values) == 2 else shap_values, X_test, show=False)
                        else:
                            shap.summary_plot(shap_values, X_test, show=False)
                        
                        st.pyplot(plt.gcf())
                        plt.clf() # 次の描画のためにリセット
                        
                    except Exception as e:
                        st.error(f"SHAP値の計算中にエラーが発生しました。\n\n詳細: {e}")

    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました。Parquet形式のファイルか確認してください。\n\n詳細なエラー: {e}")

else:
    st.info("サイドバーからParquet形式のデータをアップロードしてください。")
