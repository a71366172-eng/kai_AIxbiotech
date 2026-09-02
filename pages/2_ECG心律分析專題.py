from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="ECG 心律分析專題",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.subheader("專題頁面切換")
nav_left, nav_right = st.columns(2)
nav_left.markdown(
    '<a href="./" target="_self" style="display:block;padding:.55rem 1rem;'
    'border:1px solid #d0d7de;border-radius:.5rem;text-align:center;'
    'text-decoration:none;font-weight:600;">🧬 乳房腫瘤分類專題</a>',
    unsafe_allow_html=True,
)
nav_right.markdown(
    '<a href="./ECG心律分析專題" target="_self" style="display:block;padding:.55rem 1rem;'
    'border:1px solid #d0d7de;border-radius:.5rem;text-align:center;'
    'text-decoration:none;font-weight:600;">💓 ECG 心律分析專題</a>',
    unsafe_allow_html=True,
)
st.divider()

st.title("ECG 心律分析專題")
st.caption("獨立專題主頁｜合成 ECG 訊號｜學習用途，不是醫療檢測工具")

st.header("心電訊號與 R 峰偵測")
st.write("以可重現的合成 ECG 波形示範取樣、雜訊、R 峰偵測及心率估計。")

kind = st.selectbox("波形情境", ["規律心律", "含雜訊心律", "不規律示範"])
duration = st.slider("展示秒數", 5, 15, 8)
sample_rate = 250
time = np.arange(0, duration, 1 / sample_rate)
rng = np.random.default_rng(42)
heart_rate = {"規律心律": 72, "含雜訊心律": 78, "不規律示範": 92}[kind]
phase = (time * heart_rate / 60) % 1

signal = (
    0.12 * np.exp(-((phase - 0.18) / 0.035) ** 2)
    - 0.18 * np.exp(-((phase - 0.39) / 0.012) ** 2)
    + 1.0 * np.exp(-((phase - 0.41) / 0.010) ** 2)
    - 0.25 * np.exp(-((phase - 0.44) / 0.014) ** 2)
    + 0.28 * np.exp(-((phase - 0.68) / 0.08) ** 2)
)

if kind == "含雜訊心律":
    signal += 0.08 * np.sin(2 * np.pi * 0.7 * time) + rng.normal(0, 0.06, len(time))
elif kind == "不規律示範":
    signal += 0.04 * np.sin(2 * np.pi * 0.4 * time)
    for start in np.arange(0.8, duration, 1.3):
        signal[int(start * sample_rate): int((start + 0.05) * sample_rate)] += 0.45

ecg_frame = pd.DataFrame({"時間（秒）": time, "振幅": signal})
st.line_chart(ecg_frame.set_index("時間（秒）"), height=360)

threshold = st.slider("R 峰偵測閾值", 0.2, 1.0, 0.55, 0.05)
peaks = np.where(
    (signal[1:-1] > threshold)
    & (signal[1:-1] > signal[:-2])
    & (signal[1:-1] > signal[2:])
)[0] + 1

estimated_hr = len(peaks) / duration * 60 if len(peaks) else 0
a, b, c = st.columns(3)
a.metric("偵測峰值數", int(len(peaks)))
b.metric("估計心率（BPM）", f"{estimated_hr:.1f}")
c.metric("取樣率", f"{sample_rate} Hz")

st.subheader("分析重點")
st.markdown(
    "- 比較規律、含雜訊與不規律的示範波形\n"
    "- 調整閾值，觀察峰值偵測與估計心率的變化\n"
    "- 理解雜訊與錯誤峰值對分析結果的影響\n"
    "- 後續可加入濾波、RR interval、心律分類與跨受試者驗證"
)

st.warning("本頁使用合成訊號，不能用於判斷個人心律或任何醫療狀況。")
