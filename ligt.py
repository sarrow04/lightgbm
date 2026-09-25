import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- ダミーデータ生成機能（キャッシュして高速化） ---
@st.cache_data
def generate_dummy_csv():
    np.random.seed(42)
    n_rows = 5000
    age = np.random.randint(20, 70, n_rows)
    income = np.random.normal(500, 150, n_rows)
    store_area = np.random.normal(120, 30, n_rows)
    distance_to_station = np.random.uniform(0.1, 5.0, n_rows)
    category = np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], n_rows)
    weather = np.random.choice(['Sunny', 'Cloudy', 'Rainy'], n_rows)

    df = pd.DataFrame({
        'Age': age, 'Income': income, 'Store_Area': store_area,
        'Distance_to_Station': distance_to_station, 'Category': category, 'Weather': weather
    })

    base_sales = 1000
    sales = base_sales + (df['Income'] * 2.0) - (df['Distance_to_Station'] * 100)
    category_multiplier = {'Electronics': 1.8, 'Clothing': 1.2, 'Food': 0.7, 'Home': 1.0}
    sales *= df['Category'].map(category_multiplier)
    noise = np.random.normal(0, 300, n_rows)
    df['Sales'] = np.clip(sales + noise, 100, None)
    
    return df.to_csv(index=False).encode('utf-8')

st.title("LightGBM 手動チューニング予測アプリ")
st.write("サイドバーの数値を調整して、LightGBMの挙動と精度をリアルタイムに確認できます。")

# --- サイドバー：ダミーデータとパラメータ設定 ---
st.sidebar.header("📥 テスト用データの取得")
st.sidebar.download_button(
    label="ダミー売上データ(5000件)をダウンロード",
    data=generate_dummy_csv(),
    file_name='dummy_sales_data.csv',
    mime='text/csv',
    help="このCSVをダウンロードして、右側のアップロード画面に入れてください。"
)

st.sidebar.markdown("---")
st.sidebar.header("🔧 パラメータ手動調整")

p_max_depth = st.sidebar.slider("max_depth (木の深さ)", min_value=1, max_value=15, value=5)
p_num_leaves = st.sidebar.slider("num_leaves (葉の最大数)", min_value=2, max_value=128, value=31)
p_min_data = st.sidebar.slider("min_data_in_leaf (最小データ数)", min_value=1, max_value=100, value=20)
p_lr = st.sidebar.number_input("learning_rate (学習率)", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
p_feature_frac = st.sidebar.slider("feature_fraction (特徴量割合)", min_value=0.4, max_value=1.0, value=0.8, step=0.1)

max_leaves_limit = (2 ** p_max_depth) - 1
if p_num_leaves > max_leaves_limit:
    st.sidebar.warning(f"⚠️ max_depth={p_max_depth} の理論限界に合わせて、num_leavesを {max_leaves_limit} に自動補正します。")
    p_num_leaves = max_leaves_limit

# 1. CSVのアップロード
uploaded_file = st.file_uploader("学習用CSVデータをアップロードしてください", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("データプレビュー:", df.head())
    
    # 2. 目的変数の選択
    target_col = st.selectbox("予測したいターゲット変数を選択してください", df.columns)

    if st.button("設定したパラメータで学習を実行"):
        with st.spinner('学習を実行中です...'):
            X = df.drop(columns=[target_col])
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = X[col].astype('category')
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
            
            # 学習過程を記録するための空辞書を用意
            evals_result = {}
            
            model = lgb.train(
                params,
                train_data,
                # 訓練データと検証データの両方を渡すことで、両方の誤差推移を取得
                valid_sets=[train_data, valid_data],
                valid_names=['Train', 'Validation'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.record_evaluation(evals_result) # ここで辞書に学習過程を記録
                ],
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
            
            # --- 精度の表示 ---
            st.subheader("モデルの予測精度")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSLE", f"{rmsle:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAE", f"{mae:.4f}")
            col4.metric("R2 Score", f"{r2:.4f}")

            # --- 学習曲線の表示 ---
            st.subheader("学習曲線 (RMSEの推移)")
            # 記録した辞書からTrainとValidationのRMSEリストを取り出してデータフレーム化
            learning_curve_df = pd.DataFrame({
                'Train RMSE': evals_result['Train']['rmse'],
                'Validation RMSE': evals_result['Validation']['rmse']
            })
            st.line_chart(learning_curve_df)

            # --- SHAP値の計算と可視化 ---
            st.subheader("SHAP値による特徴量の解釈")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig)
