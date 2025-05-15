import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
from util.model_io import load_models

# ------------------ 경로 설정 ------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "model", "all_models.pkl")

# 모델 + 평가 지표 불러오기 (전체 페이지에서 사용)
model_bundle = load_models(MODEL_PATH)

st.set_page_config(page_title="학습자 이탈 예측", layout="centered")
st.title("🎓 학습자 이탈 예측 입력 폼")

st.markdown("아래 정보를 입력하면 학습자 이탈 가능성을 예측할 수 있습니다다.")

# ------------------ 입력 폼 ------------------
with st.form("dropout_form"):
    st.markdown("### 📋 학습자 정보 입력")

    # 🔹 줄 1: ID, 성별, 나이대, 장애 여부
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox("성별", ["M", "F"])
    with col2:
        age_band = st.selectbox("연령대", ["<35", "35-55", "55<="])
    with col3:
        disability = st.selectbox("장애 등록 여부", ["N", "Y"])
    with col4:
        region = st.selectbox("지역", ["East Anglian Region", "Scotland", "Wales"])
    # 🔹 줄 2: 과목 코드, 학기, 지역, 학력
    col5, col6 = st.columns(2)
    with col5:
        highest_education = st.selectbox("최종 학력", ["HE Qualification", "A Level", "Lower Than A Level"])
    with col6:
        imd_band = st.selectbox("소득구간", ["0-10%", "10-20%", "90-100%"])

    # 🔹 모델 선택 추가
    model_names = list(model_bundle.keys())

    st.markdown("### 🔍 예측 모델 선택")
    selected_model = st.selectbox("사용할 모델", model_names)

    submitted = st.form_submit_button("📊 예측하기")


# ------------------ 결과 출력 ------------------

if submitted:
    # ✅ 입력값을 DataFrame 형태로 생성
    input_data = pd.DataFrame([{
        "gender": gender,
        "region": region,
        "highest_education": highest_education,
        "imd_band": imd_band,
        "age_band": age_band,
        "disability": disability,
    }])

    # ✅ 모델 로드 및 컬럼 정렬
    model_info = model_bundle[selected_model]
    model = model_info["model"]
    X_test = model_info["metrics"]["X_test"]

    # 컬럼 순서를 학습 데이터 기준으로 맞추기
    try:
        input_data = input_data.reindex(columns=X_test.columns)
    except Exception as e:
        st.error(f"❌ 입력 데이터 정렬 실패: {e}")
        st.stop()

    # ✅ 예측 실행
    try:
        y_pred = model.predict(input_data)[0]
        y_proba = model.predict_proba(input_data)[0][1]
    except Exception as e:
        st.error(f"❌ 예측 실패: {e}")
        st.stop()

    # ✅ 결과 출력
    st.subheader("📋 입력 요약")
    st.dataframe(input_data)

    st.subheader("📈 예측 결과")
    st.success(f"✅ 예측 결과: **{'이탈' if y_pred == 1 else '유지'}**")
    st.info(f"📊 이탈 확률: **{y_proba:.2%}**")

    # ✅ 이탈 확률 게이지 차트
    import plotly.graph_objects as go
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=y_proba * 100,
        delta={"reference": 50},
        title={"text": "이탈 확률 (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "crimson" if y_proba > 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "#9be7a6"},
                {'range': [50, 75], 'color': "#ffe066"},
                {'range': [75, 100], 'color': "#ff9999"}
            ]
        }
    ))

    st.plotly_chart(gauge_fig, use_container_width=True)
