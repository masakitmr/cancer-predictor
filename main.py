import streamlit as st
import joblib
import base64
import tempfile
import numpy as np
from model_data import model_base64  # ← Base64文字列を読み込む

# --- モデル復元 ---
@st.cache_resource
def load_model():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(base64.b64decode(model_base64))
        return joblib.load(tmp.name)

model = load_model()

# --- UI ---
st.title("がん予測AIアプリ")
st.write("以下の特徴量を入力してください：")

mean_radius = st.number_input("平均半径", min_value=0.0)
mean_texture = st.number_input("平均テクスチャ", min_value=0.0)
mean_perimeter = st.number_input("平均周囲長", min_value=0.0)
mean_area = st.number_input("平均面積", min_value=0.0)

if st.button("予測する"):
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area]])
    prediction = model.predict(input_data)
    result = "悪性の可能性あり" if prediction[0] == 0 else "良性の可能性が高い"
    st.success(f"予測結果：{result}")
