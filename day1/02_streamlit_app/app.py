# app.py
import streamlit as st
import ui                   # UIモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import llm                  # LLMモジュール

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- 初期化処理 ---
metrics.initialize_nltk()
database.init_db()
data.ensure_initial_data()

# --- モデル選択UI ---
# llm.py に定義した select_model_ui() を呼び出して、
# サイドバーにモデル一覧を表示し、選択値を取得
model_name = llm.select_model_ui()

# --- モデルロード ---
# select_model_ui() で得た model_name を引数に渡す
pipe = llm.load_model(model_name)

# --- Streamlit アプリケーション ---
st.title("🤖 Gemma 2 Chatbot with Feedback")
st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
st.markdown("---")

# --- サイドバー（ナビゲーション） ---
st.sidebar.title("ナビゲーション")
if 'page' not in st.session_state:
    st.session_state.page = "チャット"
page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page),
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector)
)

# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
else:  # サンプルデータ管理
    ui.display_data_page()

# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: Your Name")
