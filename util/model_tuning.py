import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from util.visualizer import plot_confusion_matrix, plot_model_performance_comparison
from util.model_io import save_models

# 분류 모델 import
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def auto_model_tuning(base_models, param_grids, X, y, test_size=0.2, random_state=42, cv=5, n_jobs=-1):
    """
    모델 자동 튜닝 및 비교 함수

    Parameters:
    - base_models: dict (모델 이름을 키로, 초기 모델 객체를 값으로)
    - param_grids: dict (모델 이름을 키로, 파라미터 그리드를 값으로)
    - X: 특징 데이터
    - y: 타겟 데이터
    - test_size: 테스트 세트 비율 (기본 0.2)
    - random_state: 랜덤 시드 (기본 42)
    - cv: 교차 검증 폴드 수 (기본 5)
    - n_jobs: 병렬 처리 코어 수 (기본 -1: 모든 코어 사용)

    Returns:
    - results_df: 성능 비교 DataFrame
    - best_estimators: 튜닝된 최적 모델 딕셔너리
    """

    # 1. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 2. 결과 저장용 변수
    results = []
    best_estimators = {}
    # 2-1. 모델을 저장할 딕셔너리
    model_bundle = {}

    # 3. 각 모델별 그리드 서치 수행
    for model_name in base_models:
        s = time.time()
        print(f"\n >>> Tuning {model_name}...")

        grid_search = GridSearchCV(
            estimator=base_models[model_name],
            param_grid=param_grids[model_name],
            cv=cv,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # 최적 모델 저장
        best_estimators[model_name] = grid_search.best_estimator_

        # 테스트 세트 평가
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # 결과 저장
        results.append({
            'Model': model_name,
            'Best Params': grid_search.best_params_,
            'Train Accuracy (CV)': grid_search.best_score_,
            'Test Accuracy': accuracy
        })

        ## 모델 저장 구조 생성(pkl)
        model_bundle[model_name] = {
            "model": grid_search.best_estimator_,
            "metrics": {
                "best_params": grid_search.best_params_,
                "cv_score": grid_search.best_score_,
                "test_accuracy": accuracy,
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "X_test": X_test,  # ✅ 추가
                "y_test": y_test   # ✅ 추가
            }
        }

        e = time.time()
        tun_time = e-s
        print(f"- Complete:{tun_time:.5f}초")
    
    # 3-1. 모델 저장
    save_models(model_bundle, path="../model/all_models.pkl")

    # 4. 결과 DataFrame 생성
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Test Accuracy', ascending=False)
    print("\n=== Final Comparison ===")
    print(results_df.to_string(index=False))

    # 5. 시각화
    plot_model_performance_comparison(results_df)

    # 6. 성능 리포트 출력
    for model_name, estimator in best_estimators.items():
        print(f"\n {model_name} Best Model Report")
        y_pred = estimator.predict(X_test)
        print(classification_report(y_test, y_pred))

        # 혼동 행렬 시각화
        plot_confusion_matrix(y_test, y_pred, title=model_name)
        
    return results_df, best_estimators