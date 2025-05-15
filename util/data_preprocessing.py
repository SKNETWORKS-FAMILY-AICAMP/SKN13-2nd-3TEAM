import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# 경로 상수 정의
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "merged_dataset_ver.1.csv")
FINAL_OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "final_dataset.csv")


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """이상치 및 논리적 오류 행 제거"""
    df = df[~((df["final_result"] == "Fail") & (df["date_unregistration"].notnull()))]
    df = df[~((df["date_registration"].isna()) & (df["final_result"] == "Withdrawn"))]
    df["date_unregistration"] = df["date_unregistration"].fillna(9999)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """범주형 수치화 처리"""
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

    imd_order = {
        "0-10%": 1, "10-20": 2, "20-30%": 3, "30-40%": 4, "40-50%": 5,
        "50-60%": 6, "60-70%": 7, "70-80%": 8, "80-90%": 9, "90-100%": 10
    }

    # df["highest_education"] = df["highest_education"].str.strip().str.title().replace(education_order).astype(int)
    # df["age_band"] = df["age_band"].str.strip().str.title().replace(age_map).astype(int)
    # df["imd_band"] = df["imd_band"].str.strip().str.title().replace(imd_order).astype(int)

    df["highest_education"] = (
        df["highest_education"].str.strip().str.title()
        .replace(education_order).astype("Int64")
    )

    df["age_band"] = (
        df["age_band"].str.strip().str.title()
        .replace(age_map).astype("Int64")
    )

    df["imd_band"] = (
        df["imd_band"].str.strip().str.title()
        .replace(imd_order).astype("Int64")
    )


    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """ColumnTransformer 기반 결측치 처리"""
    processed_cols = ["imd_band", "date_registration"]
    passthrough_cols = [col for col in df.columns if col not in processed_cols]

    transformer = ColumnTransformer([
        ("cat_imputer", SimpleImputer(strategy="most_frequent"), ["imd_band"]),
        ("num_imputer", SimpleImputer(strategy="mean"), ["date_registration"]),
    ], remainder="passthrough")

    df_out = pd.DataFrame(transformer.fit_transform(df),
                          columns=processed_cols + passthrough_cols)
    # 원래 컬럼 순서 유지
    df_out = df_out[df.columns]
    return df_out


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    """target 열 생성 및 final_result 제거"""
    df["target"] = df["final_result"].map({
        "Pass": 0, "Distinction": 0, "Fail": 1, "Withdrawn": 1
    })

    # 예외 처리
    if df["target"].isna().any():
        unmapped = df[df["target"].isna()]
        print("⚠️ 변환되지 않은 값 존재:", unmapped["final_result"].unique())

    return df.drop(columns=["final_result"])


def preprocess_data() -> pd.DataFrame:
    """전체 전처리 파이프라인 실행 및 결과 CSV 저장"""
    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_outliers(df)
    df = encode_categorical(df)
    df = fill_missing_values(df)
    df = map_target(df)

    # 필요없는 컬럼 삭제 후 저장
    if "banked_ratio" in df.columns:
        df = df.drop(columns=["banked_ratio"])

    os.makedirs(os.path.dirname(FINAL_OUTPUT_PATH), exist_ok=True)
    df.to_csv(FINAL_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 전처리 완료. 저장 위치: {FINAL_OUTPUT_PATH}")
    return df
