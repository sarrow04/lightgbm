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
st.caption("ZIP内のParquetを読み込む独立した簡易調査アプリです。")

# ---------- ZIP / Parquet 読み込み ----------
@st.cache_data
def load_zip_parquets(file_bytes):
    found = {}
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        files = [n for n in z.namelist()
                 if not n.endswith("/") and Path(n).suffix.lower() in [".parquet", ".pq"]]
        for name in files:
            found[name] = pd.read_parquet(io.BytesIO(z.read(name)))
    return found

def make_model(X, complexity, seed):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="欠損")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))
        ]), categorical))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    if complexity == "標準":
        params = dict(objective="regression_l1", n_estimators=150,
                      learning_rate=0.03, num_leaves=10, max_depth=4,
                      min_child_samples=15)
    else:
        params = dict(objective="regression_l1", n_estimators=200,
                      learning_rate=0.03, num_leaves=15, max_depth=5,
                      min_child_samples=10)
    params.update(random_state=int(seed), n_jobs=1, verbosity=-1)
    return Pipeline([("preprocess", pre), ("model", LGBMRegressor(**params))])

# ---------- 設定ファイルの読み込み ----------
st.sidebar.header("設定の保存・読み込み")
config_file = st.sidebar.file_uploader("保存した設定JSONを読み込む", type=["json"])
if config_file is not None and st.sidebar.button("設定を読み込む"):
    try:
        loaded = json.load(config_file)
        st.session_state["loaded_config"] = loaded
        st.sidebar.success("設定を読み込みました。画面の設定に反映します。")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"設定ファイルを読み込めません: {e}")

saved = st.session_state.get("loaded_config", {})
with st.sidebar:
    st.header("学習設定")
    target = st.text_input("目的変数", value=saved.get("target", "y"), key="target_input")
    date_col = st.text_input("日付列（なければ空欄）", value=saved.get("date_column", "datetime"), key="date_input")
    cv_type = st.selectbox("検証方法", ["時系列順", "ランダム"],
                           index=["時系列順", "ランダム"].index(saved.get("validation", "時系列順"))
                           if saved.get("validation", "時系列順") in ["時系列順", "ランダム"] else 0)
    n_splits = st.slider("分割数", 2, 5, int(saved.get("folds", 3)))
    complexity = st.selectbox("LightGBM設定", ["標準", "やや複雑"],
                              index=0 if saved.get("lightgbm_setting", "標準") == "標準" else 1)
    seed = st.number_input("乱数シード", min_value=0, max_value=999999,
                           value=int(saved.get("random_seed", 42)))

st.subheader("1. ZIPファイル")
zip_file = st.file_uploader("train.parquet / test.parquet を含むZIPを選択", type=["zip"])
if zip_file is None:
    st.info("ZIP内に train.parquet（必須）と test.parquet（任意）を入れてください。")
    st.stop()

try:
    parquet_files = load_zip_parquets(zip_file.getvalue())
except Exception as e:
    st.error(f"ZIPを読み込めませんでした: {e}")
    st.stop()

if not parquet_files:
    st.error("ZIP内にParquetファイルがありません。")
    st.stop()

names = list(parquet_files.keys())
st.write("ZIP内のParquet:", " / ".join(names))

def default_file(keyword):
    matches = [n for n in names if keyword in Path(n).stem.lower()]
    return matches[0] if matches else names[0]

train_default = default_file("train")
test_matches = [n for n in names if "test" in Path(n).stem.lower()]
train_name = st.selectbox("学習データ", names, index=names.index(train_default))
test_options = ["使用しない"] + names
test_default = test_matches[0] if test_matches else "使用しない"
test_name = st.selectbox("予測データ（任意）", test_options,
                         index=test_options.index(test_default))

train_df = parquet_files[train_name].copy()
test_df = None if test_name == "使用しない" else parquet_files[test_name].copy()

st.write(f"train：{len(train_df):,}行 × {len(train_df.columns):,}列")
if test_df is not None:
    st.write(f"test：{len(test_df):,}行 × {len(test_df.columns):,}列")

with st.expander("データ確認・列情報", expanded=True):
    st.dataframe(train_df.head(10), use_container_width=True)
    col_info = pd.DataFrame({
        "列名": train_df.columns,
        "型": [str(x) for x in train_df.dtypes],
        "欠損数": train_df.isna().sum().values,
        "欠損率(%)": (train_df.isna().mean().values * 100).round(1),
        "ユニーク数": train_df.nunique(dropna=True).values
    })
    st.dataframe(col_info, use_container_width=True)

if target not in train_df.columns:
    st.error(f"目的変数「{target}」が見つかりません。")
    st.stop()
if date_col and date_col not in train_df.columns:
    st.warning(f"日付列「{date_col}」が見つからないため、日付列なしで進めます。")
    date_col = ""

features_all = [c for c in train_df.columns if c != target and c != date_col]
if test_df is not None:
    features_all = [c for c in features_all if c in test_df.columns]

# セッション内の特徴量選択を保持。設定JSONから復元した列が存在する場合は反映。
if "selected_features" not in st.session_state:
    restored = saved.get("features", features_all)
    st.session_state["selected_features"] = [c for c in restored if c in features_all]
    if not st.session_state["selected_features"]:
        st.session_state["selected_features"] = features_all
else:
    st.session_state["selected_features"] = [
        c for c in st.session_state["selected_features"] if c in features_all
    ]

st.subheader("2. 使用する特徴量")
selected = st.multiselect("学習に使う列", features_all,
                          key="selected_features")
if not selected:
    st.warning("特徴量を1列以上選択してください。")
    st.stop()

# 前回の重要度がある場合、マイナス重要度を一括で選択解除
previous_importance = st.session_state.get("importance")
if previous_importance is not None:
    negative_features = previous_importance.loc[
        previous_importance["重要度平均"] < 0, "特徴量"
    ].tolist()
    if negative_features:
        st.warning(f"マイナス重要度の特徴量：{len(negative_features)}列")
        st.caption("一括除外は学習対象から外すだけで、元データやParquetファイルは変更しません。")
        if st.button("マイナス重要度を一括除外"):
            st.session_state["selected_features"] = [
                c for c in st.session_state["selected_features"]
                if c not in negative_features
            ]
            st.session_state.pop("importance", None)
            st.session_state.pop("scores", None)
            st.session_state.pop("prediction", None)
            st.success(f"{len(negative_features)}列を学習対象から外しました。")
            st.rerun()

st.subheader("3. LightGBM検証")
if st.button("検証を実行", type="primary", use_container_width=True):
    selected = st.session_state["selected_features"]
    if not selected:
        st.error("特徴量を選択してください。")
        st.stop()

    work = train_df[selected + [target] + ([date_col] if date_col else [])].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target]).reset_index(drop=True)
    if len(work) <= n_splits:
        st.error("データ行数が少なすぎます。分割数を減らしてください。")
        st.stop()

    if date_col:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    if cv_type == "時系列順":
        if date_col:
            work = work.sort_values(date_col, kind="stable").reset_index(drop=True)
        splitter = TimeSeriesSplit(n_splits=n_splits)
        splits = list(splitter.split(work[selected]))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        splits = list(splitter.split(work[selected]))

    X, y = work[selected], work[target]
    score_rows = []
    for fold, (tr, va) in enumerate(splits, 1):
        model = make_model(X.iloc[tr], complexity, seed)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        score_rows.append({
            "fold": fold,
            "MAE": mean_absolute_error(y.iloc[va], pred),
            "RMSE": np.sqrt(mean_squared_error(y.iloc[va], pred)),
            "R2": r2_score(y.iloc[va], pred) if len(va) > 1 else np.nan
        })

    scores = pd.DataFrame(score_rows)
    st.session_state["scores"] = scores
    st.session_state["last_run_features"] = selected.copy()
    st.session_state["settings"] = {
        "train_file": train_name, "test_file": test_name,
        "target": target, "date_column": date_col,
        "features": selected, "validation": cv_type,
        "folds": n_splits, "lightgbm_setting": complexity,
        "random_seed": int(seed), "created_at": datetime.now().isoformat()
    }

    st.markdown("#### 検証スコア")
    st.dataframe(scores, use_container_width=True)
    avg = scores[["MAE", "RMSE", "R2"]].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("平均 MAE", f"{avg['MAE']:.4f}")
    c2.metric("平均 RMSE", f"{avg['RMSE']:.4f}")
    c3.metric("平均 R²", f"{avg['R2']:.4f}" if pd.notna(avg["R2"]) else "算出不可")

    # 最後の検証区間を使ったPermutation Importance
    tr, va = splits[-1]
    imp_model = make_model(X.iloc[tr], complexity, seed)
    imp_model.fit(X.iloc[tr], y.iloc[tr])
    with st.spinner("特徴量重要度を計算中..."):
        pi = permutation_importance(
            imp_model, X.iloc[va], y.iloc[va],
            scoring="neg_mean_absolute_error", n_repeats=5,
            random_state=int(seed), n_jobs=1
        )
    importance = pd.DataFrame({
        "特徴量": selected,
        "重要度平均": pi.importances_mean,
        "重要度標準偏差": pi.importances_std
    }).sort_values("重要度平均", ascending=False).reset_index(drop=True)
    st.session_state["importance"] = importance

    st.markdown("#### 特徴量重要度")
    st.caption("Permutation Importance。マイナス値は、その検証区間では列を使わない方がMAEが良かった可能性があります。")
    st.dataframe(importance, use_container_width=True)
    st.bar_chart(importance.set_index("特徴量")["重要度平均"])

    final_model = make_model(X, complexity, seed)
    final_model.fit(X, y)
    if test_df is not None:
        prediction = final_model.predict(test_df[selected])
        output = pd.DataFrame({"prediction": prediction})
        if date_col and date_col in test_df.columns:
            output.insert(0, date_col, test_df[date_col].values)
        st.session_state["prediction"] = output

# ---------- 保存・ダウンロード ----------
st.subheader("4. 設定・結果の保存")
current_settings = {
    "train_file": train_name,
    "test_file": test_name,
    "target": target,
    "date_column": date_col,
    "features": st.session_state.get("selected_features", []),
    "validation": cv_type,
    "folds": n_splits,
    "lightgbm_setting": complexity,
    "random_seed": int(seed)
}
st.download_button(
    "現在の設定をJSON保存",
    json.dumps(current_settings, ensure_ascii=False, indent=2).encode("utf-8"),
    "lightgbm_settings.json",
    "application/json",
    use_container_width=True
)
st.caption("JSONには特徴量の選択と設定を保存します。次回はサイドバーからJSONを読み込めます。")

if "scores" in st.session_state:
    st.download_button("スコアCSVを保存",
        st.session_state["scores"].to_csv(index=False).encode("utf-8-sig"),
        "lightgbm_scores.csv", "text/csv")
if "importance" in st.session_state:
    st.download_button("重要度CSVを保存",
        st.session_state["importance"].to_csv(index=False).encode("utf-8-sig"),
        "feature_importance.csv", "text/csv")
if "prediction" in st.session_state:
    st.download_button("予測CSVを保存",
        st.session_state["prediction"].to_csv(index=False).encode("utf-8-sig"),
        "prediction_results.csv", "text/csv")
