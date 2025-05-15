import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from util.model_io import load_models
from util.visualizer import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_model_performance_comparison
)

# ------------------ 경로 설정 ------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "model", "all_models.pkl")

# 모델 + 평가 지표 불러오기 (전체 페이지에서 사용)
model_bundle = load_models(MODEL_PATH)

# ------------------ 페이지 설정 ------------------
st.set_page_config(page_title="예측모델 성능 평가", layout="wide")
st.title("📊 분류 모델 성능 평가 대시보드")

# ------------------ 모델 리스트 및 선택 UI ------------------
model_names = list(model_bundle.keys())
model_options = ['모델을 선택해주세요'] + model_names
selected_model = st.selectbox("🔽 평가할 모델을 선택하세요", model_options)

# ------------------ 기본값 선택: 전체 성능 요약 ------------------
if selected_model == "모델을 선택해주세요":
    st.markdown("### ✅ 전체 모델 성능 비교")

    # 📊 평가 지표만 추출
    metrics_data = []
    for model_name, info in model_bundle.items():
        m = info['metrics']
        metrics_data.append({
            "Model": model_name,
            "Accuracy": m["test_accuracy"],
            "Precision": m["classification_report"]["1"]["precision"],
            "Recall": m["classification_report"]["1"]["recall"],
            "F1-score": m["classification_report"]["1"]["f1-score"],
            "ROC-AUC": m.get("roc_auc", np.nan)
        })

    results_df = pd.DataFrame(metrics_data).sort_values(by="Accuracy", ascending=False)

    # 📋 표 형태 출력
    st.dataframe(results_df, use_container_width=True)

    # 📈 시각화 출력
    st.markdown("#### 📉 Accuracy 기반 성능 비교 (수평 막대 차트)")
    fig = plot_model_performance_comparison(results_df[["Model", "Accuracy"]])
    st.pyplot(fig)

    st.caption("🔍 왼쪽에서 모델을 선택하면, 아래에 해당 모델의 성능 평가 그래프가 표시됩니다.")

# ------------------ 모델 선택 시: 성능 시각화 ------------------
else:
    st.markdown(f"### 🔍 {selected_model} 모델 성능 시각화")

    model_info = model_bundle[selected_model]
    model = model_info["model"]
    metrics = model_info["metrics"]
    y_test = metrics["y_test"]
    y_pred = model.predict(metrics["X_test"])
    y_proba = model.predict_proba(metrics["X_test"])[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        fig1 = plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {selected_model}")
        st.pyplot(fig1)

    with col2:
        fig2 = plot_roc_curve(y_test, y_proba, estimator_name=selected_model, title="ROC Curve")
        st.pyplot(fig2)

        fig3 = plot_precision_recall_curve(y_test, y_proba, estimator_name=selected_model, title="Precision-Recall Curve")
        st.pyplot(fig3)

    st.caption("※ 위 결과는 선택한 모델의 테스트셋 성능에 기반합니다.")
