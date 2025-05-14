import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from util.visualizer import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)

# ------------------ 페이지 설정 ------------------
st.set_page_config(page_title="예측모델 성능 평가", layout="wide")
st.title("📊 분류 모델 성능 평가 대시보드")

# ------------------ 모델 리스트 및 선택 UI ------------------
model_list = ['모델을 선택해주세요', 'DecisionTree', 'RandomForest', 'XGBoost', 'LogisticRegression']
selected_model = st.selectbox("🔽 평가할 모델을 선택하세요", model_list)

# ------------------ 더미 데이터 생성 및 모델 학습 ------------------
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 현재는 LogisticRegression으로 고정 학습 (추후 분기 처리 가능)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ------------------ 기본값 선택: 전체 성능 요약 ------------------
if selected_model == "모델을 선택해주세요":
    st.markdown("### ✅ 전체 모델 성능 비교")
    metric_table = pd.DataFrame({
        "Model": model_list[1:],  # 첫 번째 항목 제외
        "Accuracy": [0.78, 0.82, 0.85, 0.76],
        "Precision": [0.75, 0.80, 0.84, 0.74],
        "Recall": [0.72, 0.79, 0.83, 0.70],
        "F1-score": [0.735, 0.795, 0.835, 0.72],
        "ROC-AUC": [0.80, 0.86, 0.88, 0.79]
    })
    st.dataframe(metric_table, use_container_width=True)
    st.caption("🔍 왼쪽에서 모델을 선택하면, 아래에 해당 모델의 성능 평가 그래프가 표시됩니다.")

# ------------------ 모델 선택 시: 성능 시각화 ------------------
else:
    st.markdown(f"### 🔍 {selected_model} 모델 성능 시각화")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {selected_model}")
        st.pyplot(fig1)

    with col2:
        fig2 = plot_roc_curve(y_test, y_proba, estimator_name=selected_model, title="ROC Curve")
        st.pyplot(fig2)

        fig3 = plot_precision_recall_curve(y_test, y_proba, estimator_name=selected_model, title="Precision-Recall Curve")
        st.pyplot(fig3)

    st.caption("※ 현재는 모든 모델 선택 시 동일한 LogisticRegression 결과를 보여줍니다. 향후 실제 모델로 대체 가능합니다.")
