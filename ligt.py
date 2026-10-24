import json

# 1. これまでのコードをPythonの辞書（キーワード別）にまとめる
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

# 2. HTMLのテンプレート（デザインとコピペ用のJavaScript）
html_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>特徴量エンジニアリング 辞書アプリ</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { text-align: center; color: #2c3e50; }
        .category { background: #fff; margin-bottom: 20px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .category h2 { margin-top: 0; color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        .snippet { margin-bottom: 20px; }
        .snippet-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
        .snippet-title { font-weight: bold; color: #e67e22; }
        .copy-btn { background-color: #3498db; color: white; border: none; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; transition: background 0.3s; }
        .copy-btn:hover { background-color: #2980b9; }
        pre { background: #282c34; color: #abb2bf; padding: 15px; border-radius: 6px; overflow-x: auto; font-family: "Courier New", Courier, monospace; margin: 0; font-size: 14px; line-height: 1.4; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍱 お弁当需要予測 特徴量スニペット</h1>
        <div id="content"></div>
    </div>

    <script>
        // Pythonから埋め込まれたJSONデータを読み込む
        const codeData = {json_data};
        const contentDiv = document.getElementById('content');

        // HTML要素の生成
        for (const [category, snippets] of Object.entries(codeData)) {
            let categoryHtml = `<div class="category"><h2>${category}</h2>`;
            
            snippets.forEach((snippet, index) => {
                const uniqueId = `code-${Math.random().toString(36).substr(2, 9)}`;
                categoryHtml += `
                    <div class="snippet">
                        <div class="snippet-header">
                            <span class="snippet-title">📌 ${snippet.title}</span>
                            <button class="copy-btn" onclick="copyToClipboard('${uniqueId}', this)">コピー</button>
                        </div>
                        <pre><code id="${uniqueId}">${escapeHtml(snippet.code)}</code></pre>
                    </div>
                `;
            });
            categoryHtml += `</div>`;
            contentDiv.innerHTML += categoryHtml;
        }

        // HTMLエスケープ処理
        function escapeHtml(unsafe) {
            return unsafe
                 .replace(/&/g, "&amp;")
                 .replace(/</g, "&lt;")
                 .replace(/>/g, "&gt;")
                 .replace(/"/g, "&quot;")
                 .replace(/'/g, "&#039;");
        }

        // コピー機能
        function copyToClipboard(elementId, button) {
            const codeText = document.getElementById(elementId).innerText;
            navigator.clipboard.writeText(codeText).then(() => {
                const originalText = button.innerText;
                button.innerText = "コピーしました！";
                button.style.backgroundColor = "#2ecc71";
                setTimeout(() => {
                    button.innerText = originalText;
                    button.style.backgroundColor = "#3498db";
                }, 2000);
            }).catch(err => {
                console.error('クリップボードへのコピーに失敗しました', err);
                alert('コピーに失敗しました');
            });
        }
    </script>
</body>
</html>
"""

# 3. 辞書データをJSON文字列に変換してHTMLに埋め込む
json_str = json.dumps(code_data, ensure_ascii=False)
final_html = html_template.replace("{json_data}", json_str)

# 4. HTMLファイルとして保存
file_name = "my_features_app.html"
with open(file_name, "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"✅ '{file_name}' を作成しました！ダウンロードしてブラウザで開いてください。")
