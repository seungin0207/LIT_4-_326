from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# azure info
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="64a8c866-3300-4d18-b0e4-e33d8b66c040",
    resource_group_name="jbean",
    workspace_name="LIT_apr_credit_card"
)

# MLTable 데이터 등록
my_data = Data(
    path="./data/",
    type=AssetTypes.MLTABLE,
    name="credit-card-approval",
    description="신용카드 발급 승인 예측 데이터"
)

ml_client.data.create_or_update(my_data)
print("Azure ML 업로드 완료")