# 可解釋式機器學習分類專題

以公開的 Wisconsin Diagnostic Breast Cancer（WDBC）資料為例，建立一套可重現的二元分類機器學習流程。本專題重點不在特定產業，而在示範從資料理解、前處理、模型比較、評估到互動展示的完整流程。

## 專題亮點

- 569 筆樣本、30 個數值特徵
- 比較 Logistic Regression、Random Forest、SVM 三種模型
- 使用分層交叉驗證，避免只看單次切分結果
- 同時呈現 Accuracy、Precision、Recall、F1、ROC-AUC 與混淆矩陣
- 以 Logistic Regression 係數解釋特徵對模型輸出的影響
- Streamlit 互動介面支援範例個案、閾值調整與單筆解釋
- 同一個 App 以兩個獨立主頁呈現乳癌分類與 ECG 心律分析
- 訓練與測試資料明確分離，避免資料洩漏

## 快速開始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

macOS / Linux 啟用環境請改用 `source .venv/bin/activate`。

## 專案結構

```text
.
├── app.py                 # Streamlit 多頁籤互動展示
├── train.py               # 模型比較、選模與產出 artifacts
├── biomedical_ai.py       # 資料、模型、評估共用函式
├── requirements.txt
├── tests/test_pipeline.py
└── artifacts/             # 執行 train.py 後自動建立
```

## 方法

資料由 `scikit-learn` 內建的 WDBC 複本載入，原始來源為 UCI Machine Learning Repository。標籤重新定義為二元類別 0/1，方便示範分類模型與評估指標。

1. 以 80/20 分層切分訓練與測試集。
2. 標準化只在訓練折內擬合，並封裝於 Pipeline。
3. 在訓練集做 5-fold Stratified CV，比較三個模型。
4. 以 Recall 為主要選模指標、ROC-AUC 為次要指標，示範如何依問題情境選擇模型。
5. 最後只在保留的測試集評估一次。

## 如何解讀

- **Recall**：實際為正類的樣本中，被模型找出的比例。
- **Precision**：模型判為正類的樣本中，真正為正類的比例。
- **ROC-AUC**：模型跨不同閾值區分兩類的能力。
- **Decision threshold**：調低通常可提高 Recall，但也會增加假陽性；不能用單一指標取代臨床判斷。

## 一般機器學習限制

- 資料量與特徵範圍有限，模型表現不代表其他資料分布。
- 測試集只適合估計泛化能力，不代表部署後一定能維持相同結果。
- Accuracy、Precision、Recall 之間存在取捨，應依實際錯誤成本選擇閾值。
- 仍需外部資料驗證、錯誤分析、資料漂移監測與模型再訓練策略。
- 任何模型輸出都應輔助決策，不應取代領域專家判斷。

資料來源：[UCI WDBC dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)，授權 CC BY 4.0。

## 第二個專題主頁

- **ECG 心律分析專題**：以可重現的合成波形示範取樣、雜訊、R 峰偵測與心率估計，並說明可延伸至 RR interval 特徵與心律分類。

兩個主頁均為學習展示，不使用真實病患資料，也不提供醫療建議。

