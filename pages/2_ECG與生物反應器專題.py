from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest


st.set_page_config(page_title="ECG 與生物反應器專題", page_icon="🧪", layout="wide")
st.title("ECG 與生物反應器專題")
st.caption("第二主頁｜合成資料示範｜學習用途，不是醫療檢測或製程控制工具")

ecg, bio = st.tabs(["ECG 訊號分析", "生物反應器製程分析"])

with ecg:
    st.header("ECG 心律訊號分析")
    st.write("以可重現的合成 ECG 波形示範取樣、雜訊與 R 峰偵測。")
    kind = st.selectbox("波形情境", ["規律心律", "含雜訊心律", "不規律示範"])
    duration = st.slider("展示秒數", 5, 15, 8)
    sample_rate = 250
    time = np.arange(0, duration, 1 / sample_rate)
    rng = np.random.default_rng(42)
    heart_rate = {"規律心律": 72, "含雜訊心律": 78, "不規律示範": 92}[kind]
    phase = (time * heart_rate / 60) % 1
    signal = (0.12 * np.exp(-((phase - 0.18) / 0.035) ** 2)
              - 0.18 * np.exp(-((phase - 0.39) / 0.012) ** 2)
              + 1.0 * np.exp(-((phase - 0.41) / 0.010) ** 2)
              - 0.25 * np.exp(-((phase - 0.44) / 0.014) ** 2)
              + 0.28 * np.exp(-((phase - 0.68) / 0.08) ** 2))
    if kind == "含雜訊心律":
        signal += 0.08 * np.sin(2 * np.pi * 0.7 * time) + rng.normal(0, 0.06, len(time))
    elif kind == "不規律示範":
        signal += 0.04 * np.sin(2 * np.pi * 0.4 * time)
    ecg_frame = pd.DataFrame({"時間（秒）": time, "振幅": signal})
    st.line_chart(ecg_frame.set_index("時間（秒）"), height=320)
    threshold = st.slider("R 峰偵測閾值", 0.2, 1.0, 0.55, 0.05)
    peaks = np.where((signal[1:-1] > threshold) & (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0] + 1
    a, b = st.columns(2)
    a.metric("偵測峰值數", int(len(peaks)))
    b.metric("估計心率（BPM）", f"{len(peaks) / duration * 60:.1f}")
    st.info("可延伸：濾波、RR interval 特徵、正常／異常心律分類，以及跨受試者驗證。")

with bio:
    st.header("生物反應器製程趨勢與異常分析")
    st.write("以合成時間序列示範 pH、溫度、溶氧、葡萄糖與菌體濃度的趨勢分析。")
    points = st.slider("模擬資料長度", 100, 500, 240, 20)
    contamination = st.slider("示範異常比例", 0.00, 0.08, 0.03, 0.01)
    rng = np.random.default_rng(7)
    t = np.arange(points)
    phase = t / max(points - 1, 1)
    data = pd.DataFrame({
        "時間（小時）": t * 0.5,
        "pH": 6.8 + 0.12 * np.sin(t / 25) + rng.normal(0, 0.025, points),
        "溫度（°C）": 37 + 0.15 * np.sin(t / 40) + rng.normal(0, 0.04, points),
        "溶氧（%）": 68 - 10 * phase + 2.5 * np.sin(t / 17) + rng.normal(0, 0.8, points),
        "葡萄糖（g/L）": 18 * np.exp(-2.2 * phase) + rng.normal(0, 0.25, points),
        "菌體濃度（g/L）": 0.5 + 8 / (1 + np.exp(-10 * (phase - 0.45))) + rng.normal(0, 0.12, points),
    })
    anomaly_idx = rng.choice(points, max(1, int(points * contamination)), replace=False)
    data.loc[anomaly_idx, "pH"] += rng.choice([-0.7, 0.7], len(anomaly_idx))
    data.loc[anomaly_idx, "溫度（°C）"] += rng.choice([-1.8, 1.8], len(anomaly_idx))
    features = list(data.columns[1:])
    detector = IsolationForest(contamination=max(contamination, 0.01), random_state=42)
    data["異常"] = detector.fit_predict(data[features]) == -1
    st.line_chart(data.set_index("時間（小時）")[features], height=350)
    st.metric("模型標記的異常筆數", int(data["異常"].sum()))
    st.dataframe(data.loc[data["異常"], ["時間（小時）"] + features].head(10), hide_index=True)
    st.info("可延伸：批次分群、軟感測器、預測維護、製程能力分析與異常原因追蹤。")

