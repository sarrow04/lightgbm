import io, zipfile, warnings
import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
st.set_page_config(page_title="LightGBM簡易調査", page_icon="🔎", layout="wide")
st.title("LightGBM 簡易調査アプリ")
st.caption("既存の機械学習ノートブックとは別の、特徴量とスコアを手早く確認するためのアプリです。")

def read_csv(raw):
    err=None
    for enc in ["utf-8-sig","utf-8","cp932","shift_jis"]:
        try: return pd.read_csv(io.BytesIO(raw),encoding=enc)
        except Exception as e: err=e
    raise ValueError(f"CSV読込エラー: {err}")

def load_zip(upload):
    out={}
    with zipfile.ZipFile(upload) as z:
        names=[n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
        if not names: raise ValueError("ZIP内にCSVがありません")
        for n in names:
            with z.open(n) as f: out[n]=read_csv(f.read())
    return out

def make_pipe(X, params):
    nums=X.select_dtypes(include=["number","bool"]).columns.tolist()
    cats=[c for c in X.columns if c not in nums]
    transformers=[]
    if nums:
        transformers.append(("num",SimpleImputer(strategy="median"),nums))
    if cats:
        try: enc=OneHotEncoder(handle_unknown="ignore",min_frequency=2,sparse_output=False)
        except TypeError: enc=OneHotEncoder(handle_unknown="ignore",min_frequency=2,sparse=False)
        catpipe=Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="__MISSING__")),("ohe",enc)])
        transformers.append(("cat",catpipe,cats))
    prep=ColumnTransformer(transformers,remainder="drop")
    return Pipeline([("prep",prep),("model",LGBMRegressor(**params))])

def normalize_X(df, cols):
    x=df[cols].copy()
    for c in x.columns:
        if x[c].dtype=="bool": x[c]=x[c].astype("int8")
        if x[c].dtype=="object":
            n=pd.to_numeric(x[c],errors="coerce")
            denom=x[c].notna().sum()
            if denom and n.notna().sum()/denom>=0.8: x[c]=n
    return x

with st.sidebar:
    st.header("設定")
    target=st.text_input("目的変数", "y")
    date_col=st.text_input("日付列", "datetime")
    cv=st.selectbox("検証方法",["時系列CV","ランダムCV"])
    folds=st.slider("分割数",2,5,3)
    seed=st.number_input("乱数シード",0,99999,42)
    model_kind=st.selectbox("LightGBM設定",["標準（控えめ）","少し複雑"])

up=st.file_uploader("train.csv / test.csvを含むZIP",type=["zip"])
if up is None:
    st.info("ZIPをアップロードしてください。train.csvは必須、test.csvは任意です。")
    st.stop()
try: files=load_zip(up)
except Exception as e:
    st.error(str(e)); st.stop()

names=list(files)
def guess(word):
    return next((n for n in names if word in n.lower()),names[0])
a,b=st.columns(2)
with a: train_name=st.selectbox("学習CSV",names,index=names.index(guess("train")))
with b:
    test_opts=["読み込まない"]+names
    test_guess=next((n for n in names if "test" in n.lower()),"読み込まない")
    test_name=st.selectbox("テストCSV（任意）",test_opts,index=test_opts.index(test_guess))
train=files[train_name].copy()
test=None if test_name=="読み込まない" else files[test_name].copy()

st.subheader("1. データ確認")
c1,c2,c3=st.columns(3)
c1.metric("train行数",f"{len(train):,}")
c2.metric("train列数",f"{train.shape[1]:,}")
c3.metric("trainメモリ",f"{train.memory_usage(deep=True).sum()/1024**2:.1f} MB")
with st.expander("列情報・欠損率・ユニーク数"):
    info=pd.DataFrame({"列名":train.columns,
      "型":[str(train[c].dtype) for c in train.columns],
      "欠損率(%)":[round(train[c].isna().mean()*100,2) for c in train.columns],
      "ユニーク数":[train[c].nunique(dropna=True) for c in train.columns]})
    st.dataframe(info,use_container_width=True,hide_index=True)
    st.dataframe(train.head(10),use_container_width=True)

if target not in train.columns:
    st.error(f"目的変数 {target} がありません。サイドバーで変更してください。"); st.stop()
excluded={target}
if date_col in train.columns: excluded.add(date_col)
available=[c for c in train.columns if c not in excluded]
if test is not None: available=[c for c in available if c in test.columns]
st.subheader("2. 特徴量選択")
selected=st.multiselect("使用する特徴量（初期値は全列）",available,default=available)
st.caption(f"{len(selected)} / {len(available)} 列を選択中")
if not selected: st.warning("特徴量を1列以上選択してください"); st.stop()

if model_kind=="標準（控えめ）":
    params=dict(objective="regression_l1",n_estimators=150,learning_rate=.03,num_leaves=10,max_depth=4,min_child_samples=15,random_state=int(seed),verbosity=-1,n_jobs=1)
else:
    params=dict(objective="regression_l1",n_estimators=200,learning_rate=.03,num_leaves=15,max_depth=5,min_child_samples=10,random_state=int(seed),verbosity=-1,n_jobs=1)

if cv=="時系列CV" and date_col in train.columns:
    work=train.copy()
    dt=pd.to_datetime(work[date_col],errors="coerce")
    if dt.notna().any(): work=work.assign(__sort_date=dt).sort_values("__sort_date",na_position="last").drop(columns="__sort_date")
    else: st.warning("日付を解析できないため元の行順で検証します")
else:
    work=train.copy()
    if cv=="時系列CV": st.warning("日付列がないためCSVの行順を時系列として扱います")

X=normalize_X(work,selected)
y=pd.to_numeric(work[target],errors="coerce")
ok=y.notna()
X=X.loc[ok].reset_index(drop=True); y=y.loc[ok].reset_index(drop=True)
if len(X)<=folds+1: st.error("データ数が少なすぎます"); st.stop()
splitter=TimeSeriesSplit(n_splits=int(folds)) if cv=="時系列CV" else KFold(n_splits=int(folds),shuffle=True,random_state=int(seed))
do_pi=st.checkbox("Permutation Importanceも計算する（最後の検証fold）",False)
do_pred=st.checkbox("test.csvがあれば全trainで再学習して予測する",True)

if st.button("▶ LightGBMを実行",type="primary",use_container_width=True):
    rows=[]; last_model=last_x=last_y=None
    bar=st.progress(0)
    try:
        for i,(ti,vi) in enumerate(splitter.split(X),1):
            model=make_pipe(X.iloc[ti],params)
            model.fit(X.iloc[ti],y.iloc[ti])
            pred=model.predict(X.iloc[vi])
            rows.append({"fold":i,"MAE":mean_absolute_error(y.iloc[vi],pred),
                         "RMSE":np.sqrt(mean_squared_error(y.iloc[vi],pred)),
                         "R2":r2_score(y.iloc[vi],pred) if len(vi)>1 else np.nan})
            last_model,last_x,last_y=model,X.iloc[vi],y.iloc[vi]
            bar.progress(i/int(folds))
        scores=pd.DataFrame(rows)
        means=scores[["MAE","RMSE","R2"]].mean()
        st.session_state["scores"]=scores
        st.session_state["features"]=selected
        st.session_state["params"]=params
        st.session_state.pop("importance",None); st.session_state.pop("prediction",None)
        st.success("学習・検証が完了しました")
        m1,m2,m3=st.columns(3)
        m1.metric("平均 MAE",f"{means.MAE:.4f}")
        m2.metric("平均 RMSE",f"{means.RMSE:.4f}")
        m3.metric("平均 R²",f"{means.R2:.4f}")
        st.dataframe(scores,use_container_width=True,hide_index=True)
        if do_pi:
            pi=permutation_importance(last_model,last_x,last_y,n_repeats=5,random_state=int(seed),scoring="neg_mean_absolute_error",n_jobs=1)
            imp=pd.DataFrame({"特徴量":selected,"重要度(MAE悪化量)":pi.importances_mean,
                              "標準偏差":pi.importances_std}).sort_values("重要度(MAE悪化量)",ascending=False)
            st.session_state["importance"]=imp
            st.subheader("Permutation Importance")
            st.dataframe(imp,use_container_width=True,hide_index=True)
            st.bar_chart(imp.set_index("特徴量")["重要度(MAE悪化量)"])
        if do_pred and test is not None:
            final=make_pipe(X,params); final.fit(X,y)
            xt=normalize_X(test,selected)
            p=final.predict(xt)
            result=pd.DataFrame({"prediction":p})
            if date_col in test.columns: result.insert(0,date_col,test[date_col].values)
            st.session_state["prediction"]=result
            st.subheader("test予測（先頭10行）")
            st.dataframe(result.head(10),use_container_width=True)
    except Exception as e: st.exception(e)

if "scores" in st.session_state:
    st.subheader("3. ダウンロード")
    st.download_button("CVスコアCSV",st.session_state["scores"].to_csv(index=False).encode("utf-8-sig"),"lightgbm_cv_scores.csv","text/csv")
    if "importance" in st.session_state:
        st.download_button("重要度CSV",st.session_state["importance"].to_csv(index=False).encode("utf-8-sig"),"permutation_importance.csv","text/csv")
    if "prediction" in st.session_state:
        st.download_button("予測結果CSV",st.session_state["prediction"].to_csv(index=False).encode("utf-8-sig"),"prediction_results.csv","text/csv")
    config={"target":target,"date_column":date_col,"validation":cv,"folds":int(folds),"features":st.session_state["features"],"model_params":st.session_state["params"]}
    st.download_button("実験設定JSON",pd.Series(config).to_json(force_ascii=False,indent=2).encode("utf-8"),"experiment_settings.json","application/json")

st.divider()
st.caption("簡易調査版です。詳細な特徴量生成・大量の組み合わせ探索・H2O AutoML・コンペ固有の提出形式変換は含みません。時系列予測では原則「時系列CV」を使用してください。")
