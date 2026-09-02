from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from biomedical_ai import evaluate, logistic_contributions
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
KEY_FEATURES = [
    "worst radius",
    "worst perimeter",
    "worst area",
    "worst concavity",
    "worst concave points",
    "mean concave points",
]


def ensure_artifacts() -> None:
    if not (ARTIFACT_DIR / "model.joblib").exists():
        from train import main

        main()


@st.cache_resource
def load_artifacts():
    ensure_artifacts()
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    split = joblib.load(ARTIFACT_DIR / "split.joblib")
    comparison = pd.read_csv(ARTIFACT_DIR / "model_comparison.csv")
    ranges = pd.read_csv(ARTIFACT_DIR / "feature_ranges.csv", index_col=0)
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    return model, split, comparison, ranges, metrics


st.set_page_config(page_title="生醫 AI｜乳房腫瘤分類", page_icon="🧬", layout="wide")
model, split, comparison, ranges, baseline_metrics = load_artifacts()

st.title("可解釋式乳房腫瘤 AI 分類")
st.caption("WDBC 公開資料｜學習與作品展示｜不可作為醫療診斷或治療依據")

overview, demo, ecg_page, bioreactor_page, evaluation, limitations = st.tabs(
    ["專題總覽", "互動預測", "ECG 訊號分析", "生物反應器", "模型評估", "限制與倫理"]
)

with overview:
    st.subheader("問題定義")
    st.write(
        "使用 30 個由乳房腫塊細針抽吸影像計算出的細胞核特徵，分類良性與惡性。"
        "本專題把惡性設為正類，因為漏判惡性是需要明確觀察的錯誤。"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("總樣本", len(split.X_train) + len(split.X_test))
    c2.metric("輸入特徵", split.X_train.shape[1])
    c3.metric("測試 ROC-AUC", f"{baseline_metrics['roc_auc']:.3f}")
    c4.metric("惡性 Recall", f"{baseline_metrics['recall_malignant']:.3f}")
    st.subheader("設計重點")
    st.markdown(
        "- Pipeline 內標準化，避免測試資料資訊洩漏\n"
        "- 分層 5-fold 交叉驗證比較三種模型\n"
        "- 不只呈現 Accuracy，也檢視惡性 Recall、Precision、F1 與漏判數\n"
        "- 選用可解釋的 Logistic Regression 作互動展示"
    )

with demo:
    st.subheader("輸入一筆示範資料")
    preset = st.radio("起始範例", ["中位數", "測試集良性範例", "測試集惡性範例"], horizontal=True)
    if preset == "測試集良性範例":
        base = split.X_test.loc[split.y_test == 0].iloc[0].copy()
    elif preset == "測試集惡性範例":
        base = split.X_test.loc[split.y_test == 1].iloc[0].copy()
    else:
        base = split.X_train.median().copy()

    st.info("為保持介面易讀，只開放六個重要特徵；其他特徵使用所選範例值。輸出為模型分數，不是個人罹癌機率。")
    row = base.copy()
    cols = st.columns(2)
    for index, feature in enumerate(KEY_FEATURES):
        lo, hi = float(ranges.loc[feature, "min"]), float(ranges.loc[feature, "max"])
        value = float(np.clip(row[feature], lo, hi))
        row[feature] = cols[index % 2].slider(
            feature,
            min_value=lo,
            max_value=hi,
            value=value,
            step=max((hi - lo) / 200, 0.0001),
            format="%.4f",
        )
    threshold = st.slider("判為惡性的決策閾值", 0.10, 0.90, 0.50, 0.05)
    input_frame = pd.DataFrame([row], columns=split.X_train.columns)
    probability = float(model.predict_proba(input_frame)[0, 1])
    predicted = "惡性" if probability >= threshold else "良性"
    left, right = st.columns([1, 2])
    left.metric("模型分類", predicted)
    left.metric("惡性分類分數", f"{probability:.1%}")
    contributions = logistic_contributions(model, input_frame).head(10).sort_values("contribution")
    colors = ["#168aad" if x < 0 else "#d1495b" for x in contributions["contribution"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(contributions["feature"], contributions["contribution"], color=colors)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("Contribution to malignant log-odds (red: higher; blue: lower)")
    right.pyplot(fig, width="stretch")

with ecg_page:
    st.subheader("ECG 心律訊號分析（示範頁）")
    st.caption("以可重現的合成 ECG 波形示範訊號處理；不是醫療檢測工具。")
    ecg_kind = st.selectbox("波形情境", ["規律心律", "含雜訊心律", "不規律示範"], key="ecg_kind")
    duration = st.slider("展示秒數", 5, 15, 8, key="ecg_duration")
    sample_rate = 250
    time = np.arange(0, duration, 1 / sample_rate)
    rng = np.random.default_rng(42)
    heart_rate = {"規律心律": 72, "含雜訊心律": 78, "不規律示範": 92}[ecg_kind]
    phase = (time * heart_rate / 60) % 1
    # A simple beat template: P, QRS and T-like Gaussian components.
    signal = (
        0.12 * np.exp(-((phase - 0.18) / 0.035) ** 2)
        - 0.18 * np.exp(-((phase - 0.39) / 0.012) ** 2)
        + 1.0 * np.exp(-((phase - 0.41) / 0.010) ** 2)
        - 0.25 * np.exp(-((phase - 0.44) / 0.014) ** 2)
        + 0.28 * np.exp(-((phase - 0.68) / 0.08) ** 2)
    )
    if ecg_kind == "含雜訊心律":
        signal += 0.08 * np.sin(2 * np.pi * 0.7 * time) + rng.normal(0, 0.06, len(time))
    elif ecg_kind == "不規律示範":
        signal += 0.04 * np.sin(2 * np.pi * 0.4 * time)
        for start in np.arange(0.8, duration, 1.3):
            signal[int(start * sample_rate): int((start + 0.05) * sample_rate)] += 0.45
    ecg_frame = pd.DataFrame({"time_s": time, "amplitude": signal})
    st.line_chart(ecg_frame.set_index("time_s"), height=320)
    threshold = st.slider("示範 R 峰偵測閾值", 0.2, 1.0, 0.55, 0.05, key="ecg_threshold")
    peaks = np.where((signal[1:-1] > threshold) & (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0] + 1
    estimated_hr = (len(peaks) / duration) * 60 if len(peaks) else 0
    a, b, c = st.columns(3)
    a.metric("偵測到的峰值", int(len(peaks)))
    b.metric("示範心率（BPM）", f"{estimated_hr:.1f}")
    c.metric("取樣率", f"{sample_rate} Hz")
    st.write("可延伸：加入濾波、R 峰間距（RR interval）特徵、正常／異常分類，以及跨受試者驗證。")

with bioreactor_page:
    st.subheader("生物反應器製程趨勢與異常分析（示範頁）")
    st.caption("以合成的生技製程資料示範時間序列分析與異常偵測；不代表實際生產控制。")
    points = st.slider("模擬資料長度", 100, 500, 240, 20, key="bio_points")
    contamination = st.slider("示範異常比例", 0.00, 0.08, 0.03, 0.01, key="bio_anomaly")
    rng = np.random.default_rng(7)
    t = np.arange(points)
    phase = t / max(points - 1, 1)
    bio = pd.DataFrame(
        {
            "time_h": t * 0.5,
            "pH": 6.8 + 0.12 * np.sin(t / 25) + rng.normal(0, 0.025, points),
            "temperature_C": 37.0 + 0.15 * np.sin(t / 40) + rng.normal(0, 0.04, points),
            "dissolved_oxygen_pct": 68 - 10 * phase + 2.5 * np.sin(t / 17) + rng.normal(0, 0.8, points),
            "glucose_g_L": 18 * np.exp(-2.2 * phase) + rng.normal(0, 0.25, points),
            "biomass_g_L": 0.5 + 8 / (1 + np.exp(-10 * (phase - 0.45))) + rng.normal(0, 0.12, points),
        }
    )
    anomaly_count = max(1, int(points * contamination))
    anomaly_idx = rng.choice(points, anomaly_count, replace=False)
    bio.loc[anomaly_idx, "pH"] += rng.choice([-0.7, 0.7], anomaly_count)
    bio.loc[anomaly_idx, "temperature_C"] += rng.choice([-1.8, 1.8], anomaly_count)
    features = ["pH", "temperature_C", "dissolved_oxygen_pct", "glucose_g_L", "biomass_g_L"]
    detector = IsolationForest(contamination=max(contamination, 0.01), random_state=42)
    bio["anomaly"] = detector.fit_predict(bio[features]) == -1
    st.line_chart(bio.set_index("time_h")[features], height=350)
    st.dataframe(bio.loc[bio["anomaly"], ["time_h"] + features].head(10), hide_index=True)
    st.metric("模型標記的異常筆數", int(bio["anomaly"].sum()))
    st.write("可延伸：加入批次分群、軟感測器、預測維護、製程能力分析與異常原因追蹤。")

with evaluation:
    st.subheader("交叉驗證模型比較（僅訓練集）")
    display_cols = ["model", "cv_recall_mean", "cv_precision_mean", "cv_f1_mean", "cv_roc_auc_mean"]
    st.dataframe(comparison[display_cols].style.format({c: "{:.3f}" for c in display_cols[1:]}), hide_index=True)
    threshold_eval = st.slider("測試集評估閾值", 0.10, 0.90, 0.50, 0.05, key="eval_threshold")
    metrics = evaluate(model, split.X_test, split.y_test, threshold_eval)
    a, b, c, d = st.columns(4)
    a.metric("惡性 Recall", f"{metrics['recall_malignant']:.3f}")
    b.metric("惡性 Precision", f"{metrics['precision_malignant']:.3f}")
    c.metric("F1", f"{metrics['f1_malignant']:.3f}")
    d.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    matrix = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["實際良性", "實際惡性"],
        columns=["預測良性", "預測惡性"],
    )
    st.write("混淆矩陣")
    st.dataframe(matrix)
    st.warning(f"此閾值下，保留測試集中有 {metrics['fn']} 筆惡性樣本被漏判。調低閾值通常減少漏判，但會增加誤報。")

with limitations:
    st.subheader("這個模型不能證明什麼")
    st.markdown(
        "- 569 筆歷史資料無法代表所有年齡、族群、儀器與醫院。\n"
        "- 輸入是已計算的細胞核特徵，不是端到端醫療影像模型。\n"
        "- 沒有外部驗證、前瞻性試驗、模型校準與臨床效益分析。\n"
        "- 不應輸入真實病患資料；本頁不提供診斷或治療建議。\n"
        "- 真實醫療產品還需要資安、隱私、公平性、法規與臨床流程驗證。"
    )
    st.caption("資料來源：UCI WDBC（CC BY 4.0）；程式透過 scikit-learn 內建資料副本載入。")

