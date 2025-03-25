# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

# train-test 분할 (test size 0.2, random_state 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''



####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# 1. XGBoost 모델 생성 (random_state 고정)
xgb = XGBClassifier(random_state=42, eval_metric='logloss')

# 2. 하이퍼파라미터 그리드 설정
params = {
    'max_depth': [3, 5, 7, 9, 15],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 200, 300]
}

# 3. GridSearchCV 수행 (cv=5, scoring='accuracy')
grid_cv = GridSearchCV(xgb, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
grid_cv.fit(X_train, y_train)

# 4. 최적 파라미터 및 최고 정확도 출력
print(f"Best parameters:{grid_cv.best_params_}")
print(f"Best accuracy: {grid_cv.best_score_}")

# 5. 최적 모델로 테스트 세트 평가
best_xgb = grid_cv.best_estimator_
test_acc = best_xgb.score(X_test, y_test)

# 6. Feature Importance 시각화
plt.figure(figsize=(10, 6))
importance = pd.Series(best_xgb.feature_importances_, index=X_train.columns)
sns.barplot(x=importance.index, y=importance.values, palette='viridis')

plt.title('Feature Importance', fontsize=14)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()

