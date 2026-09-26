import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import shap
import matplotlib.pyplot as plt

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
    mime='text/csv'
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
    st.sidebar.warning(f"⚠️ max_depth={p_max_depth} の限界に合わせて num_leaves を {max_leaves_limit} に補正しました。")
    p_num_leaves = max_leaves_limit

# 1. CSVのアップロード
uploaded_file = st.file_uploader("学習用CSVデータをアップロードしてください", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
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
            
            evals_result = {}
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=['Train', 'Validation'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.record_evaluation(evals_result)
                ],
                num_boost_round=1000
            )

            # --- モデルとデータの状態を保存 ---
            st.session_state['trained_model'] = model
            st.session_state['feature_names'] = X.columns
            st.session_state['X_ref'] = X.copy()
            st.session_state['evals_result'] = evals_result
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test

    # === 学習済みモデルが存在する場合の表示 ===
    if 'trained_model' in st.session_state:
        model = st.session_state['trained_model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        st.success(f"学習完了！ (RMSE: {rmse:.2f} / R2: {r2:.4f})")

        # --- 新規データの予測シミュレーター ---
        st.markdown("---")
        st.subheader("🔮 予測シミュレーター")
        st.write("数値を自由に変更して、予測結果がどう変わるか試せます。")
        
        with st.form("simulation_form"):
            input_dict = {}
            X_ref = st.session_state['X_ref']
            
            # 特徴量ごとに自動で入力フォームを作成
            for col in st.session_state['feature_names']:
                if X_ref[col].dtype.name == 'category':
                    options = X_ref[col].cat.categories.tolist()
                    input_dict[col] = st.selectbox(f"{col} (カテゴリ)", options)
                else:
                    default_val = float(X_ref[col].median())
                    input_dict[col] = st.number_input(f"{col} (数値)", value=default_val)
                    
            if st.form_submit_button("この条件で予測する"):
                # 入力された値から1行のデータフレームを作成
                input_df = pd.DataFrame([input_dict])
                
                # カテゴリ型を復元（LightGBMのエラー回避）
                for col in st.session_state['feature_names']:
                    if X_ref[col].dtype.name == 'category':
                        input_df[col] = input_df[col].astype('category')
                
                # 予測の実行
                pred_val = model.predict(input_df)[0]
                st.info(f"**算出された予測値:** {pred_val:,.2f}")

        # --- 学習曲線の表示 ---
        st.markdown("---")
        st.subheader("学習曲線")
        evals = st.session_state['evals_result']
        st.line_chart(pd.DataFrame({'Train': evals['Train']['rmse'], 'Validation': evals['Validation']['rmse']}))

        # --- SHAP値の計算と可視化 ---
        st.subheader("特徴量の重要度 (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)
