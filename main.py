import streamlit as st
import joblib
import numpy as np
import tempfile
import base64
from model_data import model_base64

# モデルを一時ファイルとして復元
@st.cache_resource
def load_model():
    try:
        decoded = base64.b64decode(model_base64)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
        return joblib.load(tmp_path)
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        raise

model = load_model()

# アプリUI
st.title("がん予測AIアプリ")
st.write("以下の特徴量を入力してください：")

mean_radius = st.number_input("平均半径", min_value=0.0)
mean_texture = st.number_input("平均テクスチャ", min_value=0.0)
mean_perimeter = st.number_input("平均周囲長", min_value=0.0)
mean_area = st.number_input("平均面積", min_value=0.0)

if st.button("予測する"):
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area]])
    prediction = model.predict(input_data)[0]
    st.success(f"予測結果：{'がんの可能性あり' if prediction == 1 else 'がんの可能性なし'}")
