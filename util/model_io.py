import os
import joblib

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
