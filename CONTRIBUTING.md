# Contributing Guidelines

환영합니다! 이 프로젝트에 기여하기 전에 다음 지침을 꼭 읽어주세요.

## 디렉토리 구조

root/
├── app.py                       # Streamlit 메인 실행 파일
├── pages/
│   ├── 1_EDA.py                # 탐색적 분석
│   └── 2_Prediction.py         # 이탈 예측 시뮬레이션
├── requirements.txt             # 의존성 파일
├── .streamlit/
│   └── config.toml              # 스트림릿 배포 설정
├── data/
│   ├── raw/
│   │   └── customer_data.csv    # 원본 데이터
│   └── processed/
│       └── churn_ready.pkl      # 전처리된 데이터 or 모델 input용
├── model/
│   ├── train_model.py           # 모델 학습 스크립트
│   └── churn_model.pkl          # 학습된 모델 파일
├── utils/
│   └── preprocessing.py         # 전처리 함수 모음
├── notebooks/
│   └── eda.ipynb                # 탐색적 데이터 분석 노트북
├── CONTRIBUTING.md
└── README.md


## 1. 커밋 메세지 규칙

feat: 새로운 기능 추가
fix: 버그 수정
refactor: 코드 리팩토링
docs: 문서 수정
style: 코드 포맷팅 (기능 변경 없음)

예) feat:[구재회] 데이터셋 csv load 기능 축가

## 2. 라이브러리 관리

### 라이브러리 설치 및 requirements.txt 업데이트
pip install <package-name>
pip freeze > requirements.txt

### 의존 라이브러리 설치
pip install -r requirements.txt