import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
from util.model_io import load_models
from util.visualizer import plot_feature_importance
from scipy.sparse import issparse



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
        age_band = st.selectbox("연령대", ['55<=', '35-55', '0-35'])
    with col3:
        disability = st.selectbox("장애 등록 여부", ["N", "Y"])
    with col4:
        region = st.selectbox("지역", ['East Anglian Region', 'Scotland', 'North Western Region',
       'South East Region', 'West Midlands Region', 'Wales',
       'North Region', 'South Region', 'Ireland', 'South West Region',
       'East Midlands Region', 'Yorkshire Region', 'London Region'])
    # 🔹 줄 2: 과목 코드, 학기, 지역, 학력
    col5, col6 = st.columns(2)
    with col5:
        highest_education = st.selectbox("최종 학력", ['HE Qualification', 'A Level or Equivalent', 'Lower Than A Level',
       'Post Graduate Qualification', 'No Formal quals'])
    with col6:
        imd_band = st.selectbox("소득구간", ['90-100%', '20-30%', '30-40%', '50-60%', '80-90%', '70-80%',
       '60-70%', '40-50%', '10-20', '0-10%'])

    # 🔹 모델 선택 추가
    model_names = list(model_bundle.keys())

    # 🔹 모델 이름과 accuracy 정렬
    sorted_models = sorted(
        model_bundle.items(),
        key=lambda item: item[1]["metrics"]["test_accuracy"],  # accuracy 기준 정렬
        reverse=True
    )

    # 🔹 이름 가공: 최고 성능 모델에 (Best) 붙이기
    model_names = []
    for i, (name, model_info) in enumerate(sorted_models):
        acc = model_info["metrics"]["test_accuracy"]
        display_name = f"{name} ({acc:.2%})"
        if i == 0:
            display_name += " 🥇 (Best)"
        model_names.append(display_name)

    # 🔹 Streamlit UI
    st.markdown("### 🔍 예측 모델 선택")
    selected_display = st.selectbox("사용할 모델", model_names)

    # 🔹 실제 모델 이름 추출 (괄호 앞까지 잘라서 원래 이름과 매칭)
    selected_model = selected_display.split(" (")[0]

    submitted = st.form_submit_button("📊 예측하기")


# ------------------ 결과 출력 ------------------

if submitted:
    import pickle
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import joblib
    import numpy as np

    # ✅ 모델 로드 및 컬럼 정렬
    model_info = model_bundle[selected_model]
    model = model_info["model"]
    X_test = model_info["metrics"]["X_test"]

    cat_columns = ['code_module', 'code_presentation', 'gender', 'region', 'highest_education',
                'imd_band', 'age_band', 'num_of_prev_attempts', 'disability']
    num_columns = ['studied_credits', 'date_registration', 'date_unregistration', 'sum_click', 'avg_score']


    # 2. 전처리기 다시 정의 (학습 때 사용한 구조와 동일하게)
    fe_transformer = ColumnTransformer([
        ("category_ohe", OneHotEncoder(handle_unknown='ignore'), cat_columns),
        ("number_scaler", StandardScaler(), num_columns)
    ])

    # 3. 기존 데이터 불러오기 (유사도 비교용)
    base_df = pd.read_csv("data/final_dataset.csv")
    X_base = base_df.drop(columns="target")


    imd_order = {
        "0-10%": 1,
        "10-20": 2,
        "20-30%": 3,
        "30-40%": 4,
        "40-50%": 5,
        "50-60%": 6,
        "60-70%": 7,
        "70-80%": 8,
        "80-90%": 9,
        "90-100%": 10
    }

    education_order = {
        "No Formal Quals": 0,
        "Lower Than A Level": 1,
        "A Level Or Equivalent": 2,
        "He Qualification": 3,
        "Post Graduate Qualification": 4
    }

    age_map = {
        "0-35": 30,
        "35-55": 45,
        "55<=": 60
    }

    new_input = pd.DataFrame([{
        "gender": gender,
        "region": region,
        "highest_education": highest_education,
        "imd_band": imd_band,
        "age_band": age_band,
        "disability": disability,
        'code_module': "BBB",
        'code_presentation': "2014J",
        "num_of_prev_attempts": "0"
    }])

    # 문자열 표준화 + 매핑 적용
    new_input["imd_band"] = new_input["imd_band"].str.strip().str.title().replace(imd_order).astype("Int64")
    new_input["highest_education"] = new_input["highest_education"].str.strip().str.title().replace(education_order).astype("Int64")
    new_input["age_band"] = new_input["age_band"].str.strip().replace(age_map).astype("Int64")


    # ✅ 입력값을 DataFrame 형태로 생성
    new_input = new_input.astype({
        'code_module': 'object',
        'code_presentation': 'object',
        'gender': 'object',
        'region': 'object',
        'highest_education': 'object',
        'imd_band': 'float64',
        'age_band': 'int64',
        'num_of_prev_attempts': 'int64',
        'disability': 'object'
    })


    X_base_cat = X_base[cat_columns]
    new_cat_input = new_input[cat_columns]

    # 원핫 인코딩만 따로 적용해서 비교용 데이터 생성
    cat_encoder = OneHotEncoder(handle_unknown='ignore')
    X_base_cat_encoded = cat_encoder.fit_transform(X_base_cat).toarray()
    new_cat_encoded = cat_encoder.transform(new_cat_input).toarray()

    # 6. 유사한 기존 데이터 1개 선택
    similarities = cosine_similarity(new_cat_encoded, X_base_cat_encoded)
    most_similar_index = np.argmax(similarities)

    # 7. 해당 인덱스의 수치형 컬럼을 사용해 입력 데이터 보완
    new_complete_input = new_input.copy()
    for col in num_columns:
        new_complete_input[col] = X_base.loc[most_similar_index, col]

    # 수치형까지 포함된 최종 입력 데이터를 원래 순서로 정렬
    new_complete_input = new_complete_input.reindex(columns=X_base.columns)

    # 8. 전처리 후 예측
    fe_transformer.fit(X_base)  # DataFrame 그대로
    X_transformed = fe_transformer.transform(new_complete_input)
    if issparse(X_transformed):
        X_transformed = X_transformed.toarray()
    
    # 예측
    prediction = model.predict(X_transformed)
    print("✅ 예측 결과:", prediction)

    # 확률 예측
    proba = model.predict_proba(X_transformed)
    # 실제 예측 클래스
    pred_class = model.predict(X_transformed)

    # 이탈률 계산
    if pred_class[0] == 0:
        dropout_prob = 1 - proba[0][0]
    else:
        dropout_prob = proba[0][1]

    print(f"📊 예측 클래스: {pred_class[0]}")
    print(f"🔥 이탈률 (확률): {dropout_prob:.4f}")


    # ✅ 결과 출력
    # st.subheader("📋 입력 요약")
    # st.dataframe(new_input)

    st.subheader("📈 예측 결과")
    st.success(f"✅ 예측 결과: **{'이탈' if pred_class[0] == 1 else '유지'}**")
    st.info(f"📊 이탈 확률: **{dropout_prob:.2%}**")

    # ✅ 이탈 확률 게이지 차트
    import plotly.graph_objects as go
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dropout_prob * 100,
        delta={"reference": 50},
        title={"text": "이탈 확률 (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "crimson" if dropout_prob > 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "#9be7a6"},
                {'range': [50, 75], 'color': "#ffe066"},
                {'range': [75, 100], 'color': "#ff9999"}
            ]
        }
    ))

    st.plotly_chart(gauge_fig, use_container_width=True)

    # 피처 이름 가져오기
    feature_names = fe_transformer.get_feature_names_out()
    # 모델 feature importance 시각화
    fig = plot_feature_importance(model, feature_names, top_n=10)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning(f"⚠️ 해당 모델은 중요도 시각화를 지원하지 않습니다.")

    
