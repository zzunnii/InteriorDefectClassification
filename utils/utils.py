import os
import random
import numpy as np
import torch
from collections import Counter
import json


def seed_everything(seed=42):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_images(root_dir, class_to_idx):
    """디렉토리에서 이미지 경로와 라벨 수집"""
    image_paths, labels = [], []
    for cls in os.listdir(root_dir):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])
    return image_paths, labels


def calculate_class_weights(labels, num_classes):
    """클래스 불균형 보정을 위한 가중치 계산"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = [
        total_samples / (num_classes * class_counts[i]) if class_counts[i] > 0 else 0.0
        for i in range(num_classes)
    ]
    return class_weights


def create_soft_label_matrix(class_names, soft_label_mapping, default_weight=0.2):
    """소프트 레이블 매트릭스 생성

    Args:
        class_names: 클래스 이름 리스트
        soft_label_mapping: 소프트 레이블 매핑 (실제 클래스 -> (혼동 클래스, 가중치) 리스트)
        default_weight: 기본 소프트 레이블 가중치

    Returns:
        soft_label_matrix: 소프트 레이블 매트릭스 (numpy 배열)
    """
    num_classes = len(class_names)
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    # 기본적으로 단위 행렬로 초기화 (one-hot 인코딩)
    soft_label_matrix = np.eye(num_classes)

    # 소프트 레이블 적용
    for source_class, target_mappings in soft_label_mapping.items():
        if source_class in class_to_idx:
            source_idx = class_to_idx[source_class]

            # 타겟 클래스와 가중치 정보 추출
            weighted_targets = []

            for target_item in target_mappings:
                if isinstance(target_item, tuple) and len(target_item) == 2:
                    # (클래스명, 가중치) 형식
                    target_class, weight = target_item
                else:
                    # 클래스명만 있는 경우 기본 가중치 적용
                    target_class = target_item
                    weight = default_weight

                if target_class in class_to_idx:
                    weighted_targets.append((class_to_idx[target_class], weight))

            if weighted_targets:
                # 소스 클래스의 원래 레이블 가중치 계산
                total_weight = sum(w for _, w in weighted_targets)
                soft_label_matrix[source_idx, source_idx] = max(0.0, 1.0 - total_weight)

                # 타겟 클래스에 소프트 레이블 가중치 할당
                for target_idx, weight in weighted_targets:
                    soft_label_matrix[source_idx, target_idx] = weight

    return soft_label_matrix


def apply_label_smoothing(soft_label_matrix, smoothing_factor=0.1):
    """라벨 스무딩 적용

    Args:
        soft_label_matrix: 소프트 레이블 매트릭스
        smoothing_factor: 라벨 스무딩 계수

    Returns:
        smoothed_matrix: 라벨 스무딩이 적용된 매트릭스
    """
    num_classes = soft_label_matrix.shape[0]
    smoothed_matrix = soft_label_matrix.copy()

    for i in range(num_classes):
        # 각 클래스의 기존 소프트 레이블 가중치를 가져옴
        row_sum = smoothed_matrix[i].sum()

        # 라벨 스무딩 적용 (기존 가중치에 1-smoothing_factor 곱하고,
        # 나머지 smoothing_factor를 균등 분배)
        smoothed_matrix[i] = smoothed_matrix[i] * (1 - smoothing_factor)
        smoothed_matrix[i] += smoothing_factor / num_classes

        # 전체 합이 1이 되도록 조정
        smoothed_matrix[i] = smoothed_matrix[i] / smoothed_matrix[i].sum() * row_sum

    return smoothed_matrix


def ensure_dir(directory):
    """디렉토리가 존재하는지 확인하고 없으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_json(data, path):
    """JSON 파일 저장"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    """JSON 파일 로드"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)