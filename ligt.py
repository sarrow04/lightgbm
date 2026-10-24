import json

# 1. 辞書データ（先ほどと同じもの）
code_data = {
    "前処理・基礎": [
        {
            "title": "日付型の自動推論と分割（mixed対応）",
            "code": """df['date_parsed'] = pd.to_datetime(df['date_string'], format='mixed')
df['year'] = df['date_parsed'].dt.year
df['month'] = df['date_parsed'].dt.month
df['day'] = df['date_parsed'].dt.day"""
        },
        {
            "title": "不要カラムの安全な削除（エラー無視）",
            "code": "df = df.drop(columns=['name', 'remarks', 'event'], errors='ignore')"
        }
    ],
    "時系列・ラグ特徴量": [
        {
            "title": "Train/Testの一括結合と分割",
            "code": """# 結合フラグをつけて縦に結合
train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], ignore_index=True)

# --- ここで特徴量エンジニアリングを実施 ---

# 再び分割
train_processed = df[df['is_train'] == 1].drop(columns=['is_train'])
test_processed = df[df['is_train'] == 0].drop(columns=['is_train', 'y'])"""
        }
    ],
    "フラグ作成（ビジネス仮説）": [
        {
            "title": "寒くて天気が悪いフラグ",
            "code": """# 雨・雪、または降水量が0より大きい
is_bad_weather = df['weather'].str.contains('雨|雪') | (df['precipitation'] > 0)
# 15度未満
is_cold = df['temperature'] < 15.0

# 両方満たす場合は1、それ以外は0
df['is_cold_and_bad_weather'] = (is_bad_weather & is_cold).astype(int)"""
        },
        {
            "title": "前日からの気温差",
            "code": """# 必ず事前に日付でソートしておくこと
df['temperature_lag1'] = df['temperature'].shift(1)
df['temperature_diff'] = df['temperature'] - df['temperature_lag1']
df['temperature_diff'] = df['temperature_diff'].fillna(0)"""
        },
        {
            "title": "キラーメニュー・イベント・給料日フラグ",
            "code": """# 特定の文字列を含むか（na=Falseで欠損値エラー回避）
df['is_curry'] = df['name'].str.contains('カレー', na=False).astype(int)

# 欠損値(NaN)を利用したフラグ
df['has_event'] = df['event'].notna().astype(int)
df['is_payday'] = df['payday'].fillna(0).astype(int)"""
        }
    ],
    "可視化（EDA）": [
        {
            "title": "全数値カラムの箱ひげ図を自動レイアウトで描画",
            "code": """import matplotlib.pyplot as plt
import seaborn as sns
import math

num_cols = df.select_dtypes(include=['number']).columns
n_cols = len(num_cols)
n_rows = math.ceil(n_cols / 3)

plt.figure(figsize=(15, 5 * n_rows))
for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, 3, i)
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()"""
        }
    ]
}

# 2. Markdown形式のテキストを生成
md_content = "# 🍱 お弁当需要予測 特徴量スニペット集\n\n"
md_content += "Kaggleや実務で使える特徴量エンジニアリングのPythonコード集です。\n\n"

for category, snippets in code_data.items():
    md_content += f"## {category}\n\n"
    for snippet in snippets:
        md_content += f"### {snippet['title']}\n"
        md_content += "```python\n"
        md_content += f"{snippet['code']}\n"
        md_content += "```\n\n"

# 3. Markdownファイルとして保存
file_name = "README.md"
with open(file_name, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"✅ '{file_name}' を作成しました！これをGitHubリポジトリにアップロードしてください。")
