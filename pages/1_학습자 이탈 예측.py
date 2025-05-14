import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="학습자 이탈 예측", layout="centered")
st.title("🎓 학습자 이탈 예측 입력 폼")

st.markdown("아래 정보를 입력하면 학습자 이탈 가능성을 예측할 수 있습니다 (현재는 예측 모델 준비 중입니다).")

# ------------------ 더미 데이터 ------------------
# 사용자 점수 (더미)
user_score = {
    "질문 응답 반응률": 0.68,        # 68%
    "포럼 참여율": 0.45,            # 전체 강의 대비 토론 참여 비율
    "콘텐츠 완료율": 0.72,         # 전체 강의 중 몇 % 완료
    "평균 과제 피드백 수": 2.3,     # 과제당 피드백 수
    "수강 유지율": 0.81,            # 중도 이탈 없이 수료한 비율
    "다음 강의 등록률": 0.56,       # 다음 학기 동일 플랫폼 등록 여부
    "평균 과제 점수": 76.4,         # 실측 가능
    "토론 참여일 수": 4.5           # 전체 강의 기간 중 참여일 평균
}

# 전체 평균 점수 (더미)
mean_scores = {
    "질문 응답 반응률": 0.68,        # 68%
    "포럼 참여율": 0.45,            # 전체 강의 대비 토론 참여 비율
    "콘텐츠 완료율": 0.72,         # 전체 강의 중 몇 % 완료
    "평균 과제 피드백 수": 2.3,     # 과제당 피드백 수
    "수강 유지율": 0.81,            # 중도 이탈 없이 수료한 비율
    "다음 강의 등록률": 0.56,       # 다음 학기 동일 플랫폼 등록 여부
    "평균 과제 점수": 76.4,         # 실측 가능
    "토론 참여일 수": 4.5           # 전체 강의 기간 중 참여일 평균
}

# ------------------ 입력 폼 ------------------
with st.form("dropout_form"):
    st.markdown("### 📋 학습자 정보 입력")

    # 🔹 줄 1: ID, 성별, 나이대, 장애 여부
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        id_student = st.text_input("학습자 ID", "11391")
    with col2:
        gender = st.selectbox("성별", ["M", "F"])
    with col3:
        age_band = st.selectbox("연령대", ["<35", "35-55", "55<="])
    with col4:
        disability = st.selectbox("장애 등록 여부", ["N", "Y"])

    # 🔹 줄 2: 과목 코드, 학기, 지역, 학력
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        code_module = st.selectbox("과목 코드", ["AAA", "BBB", "CCC"])
    with col6:
        code_presentation = st.selectbox("제공 학기", ["2013J", "2013B"])
    with col7:
        region = st.selectbox("지역", ["East Anglian Region", "Scotland", "Wales"])
    with col8:
        highest_education = st.selectbox("최종 학력", ["HE Qualification", "A Level", "Lower Than A Level"])

    # 🔹 줄 3: 소득구간, 시도 횟수, 학점, 저축 비율
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        imd_band = st.selectbox("소득구간", ["0-10%", "10-20%", "90-100%"])
    with col10:
        num_of_prev_attempts = st.slider("기존 시도 횟수", 0, 10, 0)
    with col11:
        studied_credits = st.slider("학습한 학점", 0, 300, 120, step=10)
    with col12:
        banked_ratio = st.slider("저축된 학점 비율", 0.0, 1.0, 0.0, step=0.01)

    # 🔹 줄 4: 등록일, 수강취소일, 클릭 수, 평균 점수
    col13, col14, col15, col16 = st.columns(4)
    with col13:
        date_registration = st.slider("등록일 (수업 시작일 기준)", -300, 0, -159)
    with col14:
        date_unregistration = st.slider("수강 취소일", 0, 300, 0)
    with col15:
        sum_click = st.slider("총 클릭 수", 0, 10000, 934, step=50)
    with col16:
        avg_score = st.slider("과제 평균 점수", 0.0, 100.0, 82.0, step=1.0)

    submitted = st.form_submit_button("📊 예측하기")


# ------------------ 결과 출력 ------------------
if submitted:
    input_data = pd.DataFrame([{
        "code_module": code_module,
        "code_presentation": code_presentation,
        "id_student": id_student,
        "gender": gender,
        "region": region,
        "highest_education": highest_education,
        "imd_band": imd_band,
        "age_band": age_band,
        "num_of_prev_attempts": num_of_prev_attempts,
        "studied_credits": studied_credits,
        "disability": disability,
        "date_registration": date_registration,
        "date_unregistration": date_unregistration,
        "sum_click": sum_click,
        "avg_score": avg_score,
        "banked_ratio": banked_ratio,
    }])

    st.subheader("📋 입력 요약")
    st.dataframe(input_data)

    categories = list(user_score.keys())
    user_vals = [user_score[k] for k in categories]
    mean_vals = [mean_scores[k] for k in categories]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=categories,
        x=mean_vals,
        name="전체 평균",
        orientation='h',
        marker_color='lightgray'
    ))

    fig.add_trace(go.Bar(
        y=categories,
        x=user_vals,
        name="사용자 입력",
        orientation='h',
        marker_color='green'
    ))

    fig.update_layout(
        title="항목별 만족도 비교",
        xaxis=dict(range=[0, 10]),
        barmode='overlay',  # 혹은 'group'
        height=400,
    )

    # 평균 점수 게이지 차트
    avg_user_score = sum(user_score.values()) / len(user_score)
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_user_score,
        title={"text": "사용자 평균 만족도"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 5], 'color': "#ff9999"},
                {'range': [5, 7.5], 'color': "#ffe066"},
                {'range': [7.5, 10], 'color': "#9be7a6"}
            ]
        }
    ))

    # 항목별 도넛 차트
    pie_fig = go.Figure(go.Pie(
        labels=list(user_score.keys()),
        values=list(user_score.values()),
        textinfo='label+percent',
        hole=0.4
    ))
    pie_fig.update_layout(title="항목별 만족도 비중 (사용자 입력)")

    # 상위 3개 항목 비교 인디케이터
    indicator_fig = go.Figure()
    top3 = list(user_score.items())[:3]
    for i, (k, v) in enumerate(top3):
        indicator_fig.add_trace(go.Indicator(
            mode="number+delta",
            value=v,
            delta={"reference": mean_scores[k]},
            title={"text": k},
            domain={'row': i, 'column': 0}
        ))

    indicator_fig.update_layout(
        grid={'rows': 3, 'columns': 1, 'pattern': "independent"},
        title="상위 3개 항목 사용자 점수 vs 전체 평균"
    )

    # 🔽 시각화 출력 영역
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig.update_layout(
            height=300,
            margin=dict(l=30, r=30, t=40, b=30)
        ), use_container_width=True)

        st.plotly_chart(pie_fig.update_layout(
            height=300,
            margin=dict(l=30, r=30, t=40, b=30)
        ), use_container_width=True)

    with col2:
        st.plotly_chart(gauge_fig.update_layout(
            height=300,
            margin=dict(l=30, r=30, t=40, b=30)
        ), use_container_width=True)

        st.plotly_chart(indicator_fig.update_layout(
            height=300,
            margin=dict(l=30, r=30, t=40, b=30)
        ), use_container_width=True)

    st.warning("🔧 예측 모델이 아직 연동되지 않았습니다. 추후 결과가 여기에 표시됩니다.")
