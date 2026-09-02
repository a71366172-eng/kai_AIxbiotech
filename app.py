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

st.title("乳房腫瘤分類專題")
st.caption("WDBC 公開資料｜學習與作品展示｜不可作為醫療診斷或治療依據")

overview, demo, evaluation, limitations = st.tabs(
    ["專題總覽", "互動預測", "模型評估", "限制與倫理"]
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

