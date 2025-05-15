# app.py (🏠 대시보드 홈 전용)
import streamlit as st
import pandas as pd
from util.data_preprocessing import preprocess_data

# ---------------------- 페이지 설정 ----------------------
st.set_page_config(page_title="MOOC 이탈률 예측 대시보드", layout="wide")


# ---------------------- 더미 데이터 ----------------------
dummy_data = pd.DataFrame({
    "과목명": ["Advanced Literary Theory"]*5 + ["Read Forever 101"]*5,
    "강사이름": ["Taylor Swift"]*5 + ["Bruno Mars"]*5,
    "액티비티명": [f"Lesson 0{i} Assignment {i}" for i in range(1, 6)] +
                [f"Lesson 0{i} Forum {i}" for i in range(1, 6)],
    "과제채점": [91, 90, 93, 95, 92, 83, 87, 88, 82, 82]
})

# ---------------------- 메인 ----------------------
st.sidebar.image("img/mooc_logo.png", width=80)
st.sidebar.markdown("### 온라인 교육 플랫폼 이탈률 예측 대시보드")

st.markdown("""
    <h2>🏠 대시보드 홈</h2>
    <hr style='margin-top:0'>
""", unsafe_allow_html=True)

with st.expander("📌 프로젝트 소개", expanded=True):
    st.markdown("""
    - **목표**: MOOC, 학원, 대학교 LMS 등에서 수업 중단 가능성 예측
    - **데이터 예시**: 강의 시청률, 과제 제출 여부, 포럼 참여도
    - **활용 분야**: Coursera, K-MOOC, 대학 학사관리
    """)

with st.expander("📊 데이터 소개", expanded=True):
    st.dataframe(preprocess_data(), use_container_width=True)

with st.expander("⚙️ 예측 모델 개요 (사용한 알고리즘, 주요 변수 등)", expanded=True):
    st.markdown("""
    - 사용 알고리즘: XGBoost, Random Forest, Logistic Regression
    - 주요 변수: 클릭 수, 평균 점수, 나이대, 지역, 학력 등
    - 평가 지표: Accuracy, ROC-AUC, F1-score 등
    """)