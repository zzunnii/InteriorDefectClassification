"""
인테리어 결함 분류 시스템 설정 파일
"""

import os

# 기본 경로 설정
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = r"your augemented data path"  # 실제 경로에 맞게 수정
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 모델 설정
MODELS = {
    "resnet34": {
        "display_name": "ResNet34",
        "result_dir": os.path.join(RESULTS_DIR, "resnet34"),
    },
    "efficientnet_b0": {
        "display_name": "EfficientNet B0",
        "result_dir": os.path.join(RESULTS_DIR, "efficientnet_b0"),
    },
    "densenet121": {
        "display_name": "DenseNet121",
        "result_dir": os.path.join(RESULTS_DIR, "densenet121"),
    },
    "convnext_base": {
        "display_name": "ConvNext Base",
        "result_dir": os.path.join(RESULTS_DIR, "convnext_base"),
    }
}

# 앙상블 결과 디렉토리
ENSEMBLE_DIR = os.path.join(RESULTS_DIR, "ensemble")

# 학습 설정
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "max_epochs": 200,
    "patience": 10,  # 조기 종료 인내심
    "mixup_alpha": 0.3,  # Mixup 알파 값
    "image_size": 256,  # 이미지 크기
    "num_workers": 4,  # 데이터 로딩 워커 수 (윈도우에서는 0으로 설정할 필요가 있을 수 있음)
}

# 소프트 레이블 매핑 (실제 클래스 -> (혼동 클래스, 가중치) 리스트)
'''
SOFT_LABEL_MAPPING = {
    "틈새과다": [("창틀,문틀수정", 0.2)],
    "훼손": [("오염", 0.2)],
    "면불량": [("훼손", 0.2)],
}
'''

#convnext
SOFT_LABEL_MAPPING = {
    "틈새과다": [("창틀,문틀수정", 0.2)],
    "훼손": [("오염", 0.4)],
    "면불량": [("훼손", 0.2)],
}


# 기본 소프트 레이블 가중치 (0.0 ~ 1.0) - 매핑에 가중치가 지정되지 않을 경우 사용
SOFT_LABEL_WEIGHT = 0.2

# 앙상블에 사용할 모델 이름 리스트
ENSEMBLE_MODELS = ["resnet34", "efficientnet_b0", "densenet121", "convnext_base"]

# CUDA 관련 설정
USE_CUDA = True
CUDA_VISIBLE_DEVICES = "0"  # 여러 GPU 사용 시 "0,1,2,3" 형태로 지정