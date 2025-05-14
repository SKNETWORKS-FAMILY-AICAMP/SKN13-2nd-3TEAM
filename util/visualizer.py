
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    recall_score, precision_score, f1_score, accuracy_score,
    PrecisionRecallDisplay, average_precision_score, precision_recall_curve,
    RocCurveDisplay, roc_auc_score, roc_curve
)

__version__ = 1.0

######################################
# 📊 모델 평가 지표 시각화 함수
######################################

def plot_precision_recall_curve(y_true, y_scores, estimator_name=None, title=None):
    ap = average_precision_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap, estimator_name=estimator_name)
    disp.plot()
    if title:
        plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_scores, estimator_name=None, title=None):
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name=estimator_name)
    disp.plot()
    if title:
        plt.title(title)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    if title:
        plt.title(title)
    plt.show()

######################################
# 📈 EDA 시각화 함수
######################################

def plot_final_result_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='final_result', order=['Pass', 'Distinction', 'Fail', 'Withdrawn'])
    plt.title('Final Result Distribution')
    plt.ylabel('Number of Students')
    plt.xlabel('Final Result')
    plt.show()

def plot_feature_vs_target(df, feature, bins=10):
    df['bin'] = pd.qcut(df[feature], bins, duplicates='drop')
    grouped = df.groupby('bin')['target'].mean()
    plt.figure(figsize=(10,5))
    grouped.plot(kind='bar')
    plt.title(f'Churn Rate by {feature} Quantile')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()
    df.drop(columns='bin', inplace=True)

def plot_categorical_target_rate(df, col):
    plt.figure(figsize=(8,4))
    rate = df.groupby(col)['target'].mean().sort_values()
    sns.barplot(x=rate.index, y=rate.values)
    plt.title(f'Churn Rate by {col}')
    plt.ylabel('Churn Rate')
    plt.xlabel(col)
    plt.xticks(rotation=30)
    plt.show()

def plot_boxplot(df, cols=None, title="Box Plot for Outlier Detection", colors=None):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(cols)
    if colors is None:
        colors = ['skyblue'] * n
    elif isinstance(colors, str):
        colors = [colors] * n
    elif isinstance(colors, list) and len(colors) < n:
        colors = (colors * ((n // len(colors)) + 1))[:n]
    plt.figure(figsize=(1.5 * n + 3, 6))
    box = plt.boxplot([df[col].dropna() for col in cols], patch_artist=True, labels=cols)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_churn_rate_by_category(df, target_col, category_cols, colors='skyblue'):
    overall_rate = df[target_col].mean()
    plt.figure(figsize=(len(category_cols)*4, 5))
    for i, col in enumerate(category_cols):
        plt.subplot(1, len(category_cols), i+1)
        rate = df.groupby(col)[target_col].mean().sort_values(ascending=False)
        plt.bar(rate.index, rate.values, color=colors, edgecolor='black')
        plt.axhline(overall_rate, color='red', linestyle='--', label=f'전체 이탈률: {overall_rate:.2%}')
        plt.title(f"{col}별 이탈률")
        plt.xticks(rotation=45)
        plt.ylabel("이탈률")
        plt.legend()
        plt.ylim(0, max(rate.max(), overall_rate) * 1.2)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, title="히트맵 시각화를 통한 컬럼별 상관관계 확인", annot=True, cmap='coolwarm', figsize=(12, 10)):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt=".4f", cmap=cmap, linewidths=0.5, vmin=-1, vmax=1, cbar_kws={"label": "Correlation"})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

######################################
# 🧪 테스트 코드
######################################

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=42)
    df = pd.DataFrame(X, columns=['score_mean', 'click_sum', 'click_mean', 'dummy1', 'dummy2'])
    df['target'] = y
    df['gender'] = np.random.choice(['M', 'F'], size=500)
    df['highest_education'] = np.random.choice(['HE', 'A Level', 'No Formal'], size=500)
    df['final_result'] = np.random.choice(['Pass', 'Distinction', 'Fail', 'Withdrawn'], size=500)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # 모델 평가 지표 시각화
    plot_precision_recall_curve(y_test, y_proba)
    plot_roc_curve(y_test, y_proba)
    plot_confusion_matrix(y_test, y_pred)

    # 데이터 EDA 시각화
    plot_final_result_distribution(df)
    plot_feature_vs_target(df, 'score_mean')
    plot_categorical_target_rate(df, 'gender')
    plot_boxplot(df, ['score_mean', 'click_sum'])
    plot_churn_rate_by_category(df, 'target', ['gender', 'highest_education'])
    plot_correlation_heatmap(df)
