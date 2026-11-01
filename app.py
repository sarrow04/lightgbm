import streamlit as st
import pandas as pd
import numpy as np
import json
import zipfile
import io
from pathlib import Path
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor

st.set_page_config(page_title="LightGBM簡易調査アプリ", layout="wide")
st.title("LightGBM 簡易調査アプリ")
st.caption("ZIP内のParquetを読み込み、特徴量・検証スコア・重要度を確認する簡易調査アプリです。")

def read_parquet_bytes(data):
    return pd.read_parquet(io.BytesIO(data))

@st.cache_data
def load_zip_parquets(file_bytes):
    result = {}
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        parquet_names = [
            n for n in z.namelist()
            if not n.endswith("/") and Path(n).suffix.lower() in [".parquet", ".pq"]
        ]
        for name in parquet_names:
            result[name] = read_parquet_bytes(z.read(name))
    return result

def make_model(X, complexity, seed):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="欠損")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))
        ]), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    if complexity == "標準":
        params = dict(objective="regression_l1", n_estimators=150,
                      learning_rate=0.03, num_leaves=10, max_depth=4,
                      min_child_samples=15, random_state=int(seed), n_jobs=1,
                      verbosity=-1)
    else:
        params = dict(objective="regression_l1", n_estimators=200,
                      learning_rate=0.03, num_leaves=15, max_depth=5,
                      min_child_samples=10, random_state=int(seed), n_jobs=1,
                      verbosity=-1)
    return Pipeline([("preprocess", preprocessor), ("model", LGBMRegressor(**params))])

with st.sidebar:
    st.header("設定")
    target = st.text_input("目的変数（正解列）", value="y")
    date_col = st.text_input("日付列（なければ空欄）", value="datetime")
    cv_type = st.selectbox("検証方法", ["時系列順", "ランダム"])
    n_splits = st.slider("分割数", 2, 5, 3)
    complexity = st.selectbox("LightGBM設定", ["標準", "やや複雑"])
    seed = st.number_input("乱数シード", min_value=0, max_value=999999, value=42)

st.subheader("1. ZIPファイル読み込み")
zip_file = st.file_uploader("train.parquet / test.parquet を含むZIPを選択", type=["zip"])

if zip_file is None:
    st.info("ZIP内に train.parquet を入れてください。予測する場合は test.parquet も入れます。")
    st.stop()

try:
    parquet_dict = load_zip_parquets(zip_file.getvalue())
except Exception as e:
    st.error(f"ZIPを読み込めませんでした: {e}")
    st.stop()

if not parquet_dict:
    st.error("ZIP内にParquetファイル（.parquet / .pq）が見つかりません。")
    st.stop()

st.write("ZIP内で見つかったParquetファイル")
for name, df in parquet_dict.items():
    st.write(f"- {name}：{len(df):,}行 × {len(df.columns):,}列")

names = list(parquet_dict.keys())
def find_default(keyword):
    matches = [n for n in names if keyword in Path(n).stem.lower()]
    return matches[0] if matches else names[0]

train_default = find_default("train")
test_candidates = [n for n in names if "test" in Path(n).stem.lower()]
train_name = st.selectbox("学習データ（train）", names,
                          index=names.index(train_default))
test_options = ["使用しない"] + names
test_default = test_candidates[0] if test_candidates else "使用しない"
test_name = st.selectbox("予測データ（test・任意）", test_options,
                         index=test_options.index(test_default))

train_df = parquet_dict[train_name].copy()
test_df = None if test_name == "使用しない" else parquet_dict[test_name].copy()

st.success(f"train：{train_df.shape[0]:,}行 × {train_df.shape[1]:,}列")
if test_df is not None:
    st.success(f"test：{test_df.shape[0]:,}行 × {test_df.shape[1]:,}列")

with st.expander("データ内容・列情報を確認", expanded=True):
    st.dataframe(train_df.head(10), use_container_width=True)
    info = pd.DataFrame({
        "列名": train_df.columns,
        "データ型": [str(x) for x in train_df.dtypes],
        "欠損数": train_df.isna().sum().values,
        "欠損率(%)": (train_df.isna().mean().values * 100).round(1),
        "ユニーク数": train_df.nunique(dropna=True).values
    })
    st.dataframe(info, use_container_width=True)

if target not in train_df.columns:
    st.error(f"目的変数 '{target}' が train.parquet にありません。")
    st.stop()

if date_col and date_col not in train_df.columns:
    st.warning(f"日付列 '{date_col}' が見つからないため、日付列なしとして扱います。")
    date_col = ""

available_features = [c for c in train_df.columns if c != target and c != date_col]
if test_df is not None:
    available_features = [c for c in available_features if c in test_df.columns]

st.subheader("2. 使用する特徴量")
selected = st.multiselect("学習に使う列（初期状態は全選択）",
                          options=available_features, default=available_features)
if not selected:
    st.warning("特徴量を1列以上選択してください。")
    st.stop()

st.subheader("3. 検証")
if st.button("LightGBMで検証する", type="primary", use_container_width=True):
    work = train_df[selected + [target] + ([date_col] if date_col else [])].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target]).reset_index(drop=True)
    if len(work) < n_splits + 1:
        st.error("データ行数が少なすぎます。分割数を減らしてください。")
        st.stop()

    if cv_type == "時系列順":
        if date_col:
            work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
            work = work.sort_values(date_col, kind="stable").reset_index(drop=True)
        X, y = work[selected], work[target]
        splitter = TimeSeriesSplit(n_splits=n_splits)
        splits = list(splitter.split(X))
    else:
        X, y = work[selected], work[target]
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        splits = list(splitter.split(X))

    score_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
        model = make_model(X_tr, complexity, seed)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        score_rows.append({
            "fold": fold,
            "MAE": mean_absolute_error(y_va, pred),
            "RMSE": np.sqrt(mean_squared_error(y_va, pred)),
            "R2": r2_score(y_va, pred) if len(y_va) > 1 else np.nan
        })

    scores = pd.DataFrame(score_rows)
    st.session_state["scores"] = scores
    st.session_state["settings"] = {
        "train_file": train_name, "test_file": test_name,
        "target": target, "date_column": date_col, "features": selected,
        "validation": cv_type, "folds": n_splits,
        "lightgbm_setting": complexity, "random_seed": int(seed),
        "created_at": datetime.now().isoformat()
    }

    st.subheader("検証結果")
    st.dataframe(scores, use_container_width=True)
    avg = scores[["MAE", "RMSE", "R2"]].mean()
    a, b, c = st.columns(3)
    a.metric("平均 MAE", f"{avg['MAE']:.4f}")
    b.metric("平均 RMSE", f"{avg['RMSE']:.4f}")
    c.metric("平均 R²", f"{avg['R2']:.4f}" if pd.notna(avg["R2"]) else "算出不可")

    last_train_idx, last_valid_idx = splits[-1]
    last_model = make_model(X.iloc[last_train_idx], complexity, seed)
    last_model.fit(X.iloc[last_train_idx], y.iloc[last_train_idx])
    with st.spinner("Permutation Importanceを計算中..."):
        pi = permutation_importance(
            last_model, X.iloc[last_valid_idx], y.iloc[last_valid_idx],
            scoring="neg_mean_absolute_error", n_repeats=5,
            random_state=int(seed), n_jobs=1
        )
    imp = pd.DataFrame({
        "特徴量": selected,
        "重要度平均": pi.importances_mean,
        "重要度標準偏差": pi.importances_std
    }).sort_values("重要度平均", ascending=False)
    st.session_state["importance"] = imp
    st.subheader("特徴量重要度（最後の検証区間）")
    st.caption("Permutation Importanceです。マイナス値は、その検証区間では列を使わない方がMAEが良かった可能性を示します。")
    st.dataframe(imp, use_container_width=True)
    st.bar_chart(imp.set_index("特徴量")["重要度平均"])

    final_model = make_model(X, complexity, seed)
    final_model.fit(X, y)
    if test_df is not None:
        pred_test = final_model.predict(test_df[selected])
        out = pd.DataFrame({"prediction": pred_test})
        if date_col and date_col in test_df.columns:
            out.insert(0, date_col, test_df[date_col].values)
        st.session_state["prediction"] = out
        st.subheader("test.parquet の予測結果")
        st.dataframe(out.head(20), use_container_width=True)

st.subheader("4. 結果ダウンロード")
if "scores" in st.session_state:
    st.download_button("スコアCSV", st.session_state["scores"].to_csv(index=False).encode("utf-8-sig"),
                       "lightgbm_scores.csv", "text/csv")
if "importance" in st.session_state:
    st.download_button("重要度CSV", st.session_state["importance"].to_csv(index=False).encode("utf-8-sig"),
                       "feature_importance.csv", "text/csv")
if "prediction" in st.session_state:
    st.download_button("予測CSV", st.session_state["prediction"].to_csv(index=False).encode("utf-8-sig"),
                       "prediction_results.csv", "text/csv")
if "settings" in st.session_state:
    st.download_button("設定JSON", json.dumps(st.session_state["settings"], ensure_ascii=False, indent=2).encode("utf-8"),
                       "experiment_settings.json", "application/json")
