# llm.py
import os
import torch
from transformers import pipeline
import streamlit as st
import time
from config import MODEL_NAME as DEFAULT_MODEL
from huggingface_hub import login

# モデルをキャッシュして再利用
@st.cache_resource
def load_model(selected_model: str):
    """LLMモデルをロードする"""
    try:
        # Hugging Face のトークンでログイン
        hf_token = st.secrets["huggingface"]["token"]
        login(token=hf_token)

        # 使用デバイスの判定
        device = 0 if torch.cuda.is_available() else -1
        st.info(f"Using device: {'cuda' if device == 0 else 'cpu'}")

        # パイプラインの生成
        pipe = pipeline(
            "text-generation",
            model=selected_model,
            device=device,
            torch_dtype=torch.bfloat16 if device == 0 else None,
            # trust_remote_code=True  # 必要に応じて有効化
        )

        st.success(f"モデル '{selected_model}' の読み込みに成功しました。")
        return pipe

    except Exception as e:
        st.error(f"モデル '{selected_model}' の読み込みに失敗しました: {e}")
        return None

def select_model_ui():
    """サイドバーにモデル選択UIを追加して、選択したモデル名を返す"""
    st.sidebar.header("🔄 モデル選択")
    # ここに好きなモデル名を追加してください
    model_options = [
        DEFAULT_MODEL,
        "gpt2",
        "gpt2-medium",
        "distilgpt2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]
    return st.sidebar.selectbox("使用するLLMモデル", model_options, index=model_options.index(DEFAULT_MODEL))

def generate_response(pipe, user_question):
    """LLMを使用して質問に対する回答を生成する"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()
        # text-generation パイプラインには文字列だけ渡します
        full_text = pipe(
            user_question,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )[0]["generated_text"]

        # ユーザー入力以降を抽出（プロンプトが残る場合の簡易処理）
        response = full_text.split(user_question, 1)[-1].strip()
        end_time = time.time()

        return response, end_time - start_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        return f"エラーが発生しました: {str(e)}", 0

# --- Streamlit アプリ本体での呼び出し例 ---
if __name__ == "__main__":
    st.title("LLM Chat デモ")

    # (1) モデル選択 UI
    model_name = select_model_ui()

    # (2) モデルロード
    pipe = load_model(model_name)

    # (3) 質問入力 & 回答生成
    user_input = st.text_input("あなたの質問", "")
    if user_input:
        answer, latency = generate_response(pipe, user_input)
        st.markdown(f"**回答**: {answer}")
        st.caption(f"生成時間: {latency:.2f} 秒")
