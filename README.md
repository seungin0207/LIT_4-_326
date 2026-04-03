# LIT_4-_326

사용할 데이터셋
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction

전처리한 CSV 파일
[final_approval_data.csv](https://github.com/user-attachments/files/26441555/final_approval_data.csv)

# MS_Learn 링크
https://learn.microsoft.com/ko-kr/training/modules/find-best-classification-model-automated-machine-learning/

# 소개
# 💳 나도 카드 발급될까?
### AI 기반 대안 신용평가 모델 | Azure AutoML · Python · Kaggle

---

> "취업도 안 했고, 연봉도 없는데 카드 발급이 될까?"
>
> 대학생이라면 한 번쯤 겪어봤을 그 막막함에서 시작한 프로젝트입니다.
> 기존 신용 평가 방식이 놓치는 **잠재적 우량 고객**을 데이터로 찾아내고자 했습니다.

---

## 🔍 프로젝트 배경

전통적인 신용 평가는 직장과 연봉 중심이라, 금융 이력이 부족한 청년층(Thin-filer)은 실제 신용도와 무관하게 거절되는 경우가 많습니다.

이 프로젝트는 단순 소득 유무를 넘어, **고용 안정성 + 인구통계 특성**을 결합해 신용카드 발급 승인 여부를 예측하는 대안 모델을 만들어봤습니다.

---

## 📊 데이터 & 전처리

- **Source**: [Kaggle — Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

| 전처리 항목 | 내용 |
|---|---|
| 데이터 병합 | 신청 데이터 + 결제 이력 데이터를 병합해 실제 신용도를 반영한 타겟 생성 |
| 타겟 라벨링 | 60일 이상 연체(STATUS 2~5) → 위험(1) / 우량(0) 이진 분류 |
| 피처 변환 | `DAYS_BIRTH`, `DAYS_EMPLOYED` → `Age`, `Years_Employed` 등 직관적 단위로 변환 |
| 이상치 처리 | 이상치(365,243일)를 분석해 `Unemployed` 파생 변수 생성 |
| 클래스 불균형 | 소수 클래스(위험군 1.7%) 대응 → 층화 추출(Stratified Split) 적용 |

---

## 🛠 기술 스택

| 구분 | 내용 |
|---|---|
| 환경 | Azure Machine Learning Studio, Jupyter Notebook |
| 언어 & 라이브러리 | Python, Pandas, NumPy, Scikit-learn |
| AI / ML | Azure AutoML (Classification), VotingEnsemble, LightGBM, XGBoost |
| 해석 가능성 | Explainable AI (XAI), Feature Importance |

---

## 📈 모델링 & 평가

Azure AutoML로 여러 모델을 자동 탐색했고, `AUC_weighted`를 주 지표로 삼아 클래스 불균형 환경에서도 의미 있는 성능을 냈습니다.

Feature Importance를 통해 **근속 연수, 고용 형태** 등이 신용 승인에 큰 영향을 준다는 것을 데이터로 확인했습니다.

### 탐색된 주요 모델
- `VotingEnsemble` — 최종 채택 모델
- `StackEnsemble`
- `MaxAbsScaler, LightGBM`
- `MaxAbsScaler, XGBoostClassifier`
- `MaxAbsScaler, ExtremeRandomTrees`

---

---

<p align="center">포용적 금융 · 리스크 자동화 · 데이터 기반 의사결정</p>
