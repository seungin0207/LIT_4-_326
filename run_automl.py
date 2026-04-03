# ============================================================
# 신용카드 발급 승인 예측 - Azure AutoML 실험
# Credit Card Approval Prediction - Azure AutoML Experiment
# ============================================================

from azure.ai.ml import MLClient, automl, Input
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential

# ────────────────────────────────────────
# 1. Azure ML 워크스페이스 연결
# ────────────────────────────────────────
ml_client = MLClient(
    AzureCliCredential(),
    subscription_id="64a8c866-3300-4d18-b0e4-e33d8b66c040",
    resource_group_name="jbean",
    workspace_name="LIT_apr_credit_card"
)
print("✅ Azure ML 워크스페이스 연결 완료")

# ────────────────────────────────────────
# 2. 업로드한 데이터 자산 불러오기
# ────────────────────────────────────────
my_training_data = Input(
    type=AssetTypes.MLTABLE,
    path="azureml:credit-card-approval:1"  # 이름:버전
)
print("✅ 데이터 자산 로드 완료")

# ────────────────────────────────────────
# 3. AutoML 분류 실험 구성
# ────────────────────────────────────────
classification_job = automl.classification(
    compute="wjdbeen526",               # azure ml에서 만든 컴퓨팅 사용
    experiment_name="credit-card-approval-automl",
    training_data=my_training_data,
    target_column_name="TARGET",        # 예측할 컬럼 (0: 정상, 1: 연체)
    primary_metric="AUC_weighted",      # 불균형 데이터에 적합한 메트릭
    n_cross_validations=5,              # 5-fold 교차 검증
    enable_model_explainability=True    # Feature Importance 활성화
)
print("✅ AutoML 실험 구성 완료")

# ────────────────────────────────────────
# 4. 실험 제한 설정 (비용 및 시간 절약)
# ────────────────────────────────────────
classification_job.set_limits(
    timeout_minutes=60,           # 전체 실험 최대 60분
    trial_timeout_minutes=20,     # 개별 모델 최대 20분
    max_trials=5,                 # 최대 5개 모델 시도
    enable_early_termination=True # 성능 개선 없으면 조기 종료
)
print("✅ 실험 제한 설정 완료")

# ────────────────────────────────────────
# 5. 실험 제출
# ────────────────────────────────────────
returned_job = ml_client.jobs.create_or_update(classification_job)

print("\n🚀 AutoML 실험 제출 완료!")
print(f"   - 실험명: {returned_job.name}")
print(f"   - 상태: {returned_job.status}")
print(f"\n📊 아래 링크에서 실험 진행상황을 확인하세요!")
print(f"   {returned_job.studio_url}")
