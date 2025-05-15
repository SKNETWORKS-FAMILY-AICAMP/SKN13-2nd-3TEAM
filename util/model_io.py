import os
import joblib

# 단일 모델 저장 및 로드 함수
def save_model(model, path: str):
    """
    모델을 지정된 경로에 저장합니다.
    
    Args:
        model: 학습된 모델 객체
        path (str): 저장할 파일 경로 (.pkl 확장자 권장)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[✔] 모델이 저장되었습니다: {path}")


def load_model(path: str):
    """
    지정된 경로에서 모델을 불러옵니다.
    
    Args:
        path (str): 모델 파일 경로

    Returns:
        불러온 모델 객체
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {path}")
    
    model = joblib.load(path)
    print(f"[✔] 모델이 로드되었습니다: {path}")
    return model


# 여러개의 모델과 평가 지표를 저장하는 함수
def save_models(model_bundle: dict, path: str = "models/all_models.pkl"):
    """
    모델들과 평가 지표들을 하나의 파일로 저장합니다.
    Args:
        model_bundle (dict): {"ModelName": {"model": 모델객체, "metrics": {...}}}
        path (str): 저장 경로 (.pkl)
        
        #모델 저장 예시
        model_bundle[model_name] = {
        "model": grid_search.best_estimator_,
        "metrics": {
            "best_params": grid_search.best_params_,
            "cv_score": grid_search.best_score_,
            "test_accuracy": accuracy,
            "classification_report": cls_report,
            "confusion_matrix": cm
        }
    }
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model_bundle, path)
    print(f"[✔] 전체 모델과 지표가 저장되었습니다: {path}")


def load_models(path: str = "models/all_models.pkl") -> dict:
    """
    모델들과 지표를 불러옵니다.

    Returns:
        model_bundle (dict): 저장된 모델 + 지표 딕셔너리
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    bundle = joblib.load(path)
    print(f"[✔] 전체 모델과 지표가 로드되었습니다: {path}")
    return bundle
