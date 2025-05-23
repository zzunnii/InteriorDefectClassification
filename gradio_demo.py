#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
인테리어 결함 분류 Gradio 데모
OOD Detection 기능 포함
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import gradio as gr
import io
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가 (상대 임포트 문제 해결)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 상위 디렉토리
sys.path.insert(0, project_root)


# 프로젝트 모듈 import
from config import MODELS, TRAIN_CONFIG
from utils.models import get_model_by_name
from utils.utils import ensure_dir
from utils.dataset import get_transforms

# OOD 감지를 위한 설정
CONFIDENCE_THRESHOLD = 0.6  # 이 값보다 낮은 최대 확률을 가진 샘플은 OOD로 간주
ENTROPY_THRESHOLD = 1.0  # 이 값보다 높은 엔트로피를 가진 샘플은 OOD로 간주
MODEL_NAME = "convnext_base"  # 사용할 모델
OOD_METHOD = "confidence"  # OOD 감지 방법: "confidence", "entropy"


def calculate_entropy(probs):
    """확률 분포의 엔트로피 계산"""
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    # 0에 가까운 값 처리 (로그 계산 시 문제 방지)
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))


def load_model():
    """모델 로드 함수"""
    try:
        # 메타데이터 로드
        model_config = MODELS[MODEL_NAME]
        model_dir = model_config["result_dir"]
        metadata_path = os.path.join(model_dir, "metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            class_names = metadata["class_names"]
        else:
            # 메타데이터 파일이 없는 경우 기본값 사용
            class_names = ["균열", "들뜸", "마감불량", "변색", "오염", "오타공", "울음", "탈락", "파손", "피스", "훼손"]
            print(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
            print(f"기본 클래스 이름을 사용합니다.")

        # "정상/탐지 불가" 클래스 추가 (OOD 클래스)
        class_names.append("정상/탐지 불가")

        # 모델 로드
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = len(class_names) - 1  # OOD 클래스는 모델에 없음

        # 모델 생성
        model = get_model_by_name(MODEL_NAME, num_classes).to(device)

        # 모델 파일 경로
        model_path = os.path.join(model_dir, "best_model.pth")

        if os.path.exists(model_path):
            # 모델 가중치 로드
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"모델 로드 성공: {model_path}")
        else:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
            print(f"임의 가중치를 사용합니다.")

        model.eval()

        # 변환 함수 가져오기
        _, test_transform = get_transforms(img_size=TRAIN_CONFIG["image_size"])

        return model, class_names, test_transform, device

    except Exception as e:
        # 예외 발생 시 기본값 사용
        print(f"모델 로드 중 오류 발생: {e}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_names = ["균열", "들뜸", "마감불량", "변색", "오염", "오타공", "울음", "탈락", "파손", "피스", "훼손", "정상/탐지 불가"]
        num_classes = len(class_names) - 1

        model = get_model_by_name(MODEL_NAME, num_classes).to(device)
        model.eval()

        # 기본 변환 함수
        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return model, class_names, test_transform, device


def prepare_image(image):
    """입력 이미지 전처리"""
    if image is None:
        return None

    # PIL 이미지로 변환
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # 이미지 변환
    image_tensor = test_transform(image)
    return image_tensor.unsqueeze(0)  # 배치 차원 추가


def detect_ood(outputs, probs, method=OOD_METHOD):
    """OOD 샘플 감지"""
    if method == "confidence":
        # 최대 확률 기반 OOD 감지
        max_prob = torch.max(probs).item()
        is_ood = max_prob < CONFIDENCE_THRESHOLD
        score = max_prob
        threshold = CONFIDENCE_THRESHOLD
        criterion = "신뢰도"
        better = "높을수록"

    elif method == "entropy":
        # 엔트로피 기반 OOD 감지
        ent = calculate_entropy(probs[0])
        is_ood = ent > ENTROPY_THRESHOLD
        score = ent
        threshold = ENTROPY_THRESHOLD
        criterion = "엔트로피"
        better = "낮을수록"

    else:
        raise ValueError(f"지원하지 않는 OOD 감지 방법입니다: {method}")

    return is_ood, score, threshold, criterion, better


def predict(model, image_tensor, class_names, device, ood_method=OOD_METHOD):
    """모델 예측 및 OOD 감지"""
    # 이미지 텐서를 장치로 이동
    image_tensor = image_tensor.to(device)

    # 예측
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

    # OOD 감지
    is_ood, score, threshold, criterion, better = detect_ood(
        outputs, probs, method=ood_method
    )

    # 결과 데이터 준비
    probs = probs[0].cpu().numpy()
    outputs = outputs[0].cpu().numpy()

    # 최대 확률과 예측 클래스
    pred_class = np.argmax(probs)

    # OOD 클래스 확률 추가 (원래 클래스에 대한 확률이 낮은 경우 OOD 확률이 높음)
    if is_ood:
        pred_class = len(class_names) - 1  # OOD 클래스 인덱스
        prediction = class_names[pred_class]

        # 원래 확률에 OOD 확률 추가 (1 - max_prob를 OOD 확률로 사용)
        ood_prob = 1 - np.max(probs)
        ood_probs = np.append(probs, ood_prob)
    else:
        prediction = class_names[pred_class]
        # 원래 확률에 OOD 확률 추가 (0에 가까운 값)
        ood_probs = np.append(probs, 0.01)

    return prediction, ood_probs, is_ood, score, threshold, criterion, better


def create_bar_chart(class_names, probabilities):
    """클래스별 확률 바 차트 생성"""
    # 상위 5개 클래스 (또는 모든 클래스가 5개 미만인 경우 모든 클래스)
    num_to_show = min(5, len(class_names))

    # 확률을 기준으로 인덱스 정렬
    sorted_indices = np.argsort(probabilities)[::-1][:num_to_show]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_probs = probabilities[sorted_indices]

    # 그림 생성
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # 바 차트 그리기 - 항목별로 다른 색상 사용
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_classes)))
    bars = ax.barh(sorted_classes[::-1], sorted_probs[::-1], color=colors)

    # 바 위에 확률값 표시
    for bar, prob in zip(bars, sorted_probs[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{prob:.3f}', va='center')

    # 그래프 설정
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('확률')
    ax.set_title('클래스별 예측 확률')
    fig.tight_layout()

    # 그래프를 이미지로 변환
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # PIL 이미지로 변환
    img = Image.open(buf)
    return img


def create_confidence_gauge(score, threshold, criterion, better):
    """OOD 판단 기준 게이지 차트 생성"""
    fig = Figure(figsize=(6, 2))
    ax = fig.add_subplot(111)

    # 게이지 범위 설정
    min_val, max_val = 0, 1
    if criterion == "엔트로피":
        min_val, max_val = 0, 2.5

    # 게이지 배경 그리기
    if better == "높을수록":
        colors = [(0.9, 0.2, 0.2), (0.9, 0.6, 0.3), (0.2, 0.7, 0.3)]
        cmap = plt.cm.RdYlGn
    else:  # 낮을수록
        colors = [(0.2, 0.7, 0.3), (0.9, 0.6, 0.3), (0.9, 0.2, 0.2)]
        cmap = plt.cm.RdYlGn_r

    # 게이지 배경 그리기
    for i, color in enumerate(colors):
        pos = min_val + (max_val - min_val) * i / len(colors)
        width = (max_val - min_val) / len(colors)
        ax.barh(0, width, left=pos, height=0.5, color=color, alpha=0.6)

    # 임계값 표시
    ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label=f'임계값: {threshold:.2f}')

    # 현재 값 화살표 표시
    ax.scatter(score, 0, color='blue', s=150, zorder=5, marker='^', label=f'현재 값: {score:.2f}')

    # 그래프 설정
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(f'{criterion} ({better} 정상)')
    ax.set_title(f'OOD 판단 기준 ({criterion})')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    fig.tight_layout()

    # 그래프를 이미지로 변환
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # PIL 이미지로 변환
    img = Image.open(buf)
    return img


def process_image(input_image, ood_method):
    """입력 이미지 처리 및 결과 반환 (Gradio 인터페이스 함수)"""
    # OOD 감지 방법 설정
    global OOD_METHOD
    OOD_METHOD = ood_method

    # 이미지 준비
    image_tensor = prepare_image(input_image)
    if image_tensor is None:
        return "이미지를 업로드해주세요.", None, None

    # 예측
    prediction, probabilities, is_ood, score, threshold, criterion, better = predict(
        model, image_tensor, class_names, device, ood_method=ood_method
    )

    # OOD 감지 결과 메시지
    if is_ood:
        result_message = f"결과: {prediction} (OOD 감지됨 - {criterion} {score:.3f})"
    else:
        result_message = f"결과: {prediction} ({criterion} {score:.3f})"

    # 확률 바 차트 생성
    prob_chart = create_bar_chart(class_names, probabilities)

    # 신뢰도 게이지 생성
    gauge_chart = create_confidence_gauge(score, threshold, criterion, better)

    return result_message, prob_chart, gauge_chart


# 필요한 라이브러리 가져오기
try:
    from torchvision import transforms
except ImportError:
    import torch.nn as nn


    class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, x):
            return self.func(x)


    class Transforms:
        def __init__(self):
            pass

        def Compose(self, transforms_list):
            return transforms_list

        def Resize(self, size):
            return Lambda(lambda x: x)

        def ToTensor(self):
            return Lambda(lambda x: x)

        def Normalize(self, mean, std):
            return Lambda(lambda x: x)


    transforms = Transforms()

# 모델 및 메타데이터 로드
print("모델 로드 중...")
model, class_names, test_transform, device = load_model()
print(f"모델 로드 완료: {MODEL_NAME}")
print(f"클래스: {class_names}")
print(f"장치: {device}")

# Gradio 인터페이스 생성
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="입력 이미지"),
        gr.Radio(
            choices=["confidence", "entropy"],
            value="confidence",
            label="OOD 감지 방법"
        )
    ],
    outputs=[
        gr.Textbox(label="예측 결과"),
        gr.Image(type="pil", label="클래스별 확률"),
        gr.Image(type="pil", label="OOD 판단 기준")
    ],
    title="인테리어 결함 분류 (OOD Detection 적용)",
    description=f"""
        인테리어 결함을 분류하는 AI 모델입니다. 결함이 확실하지 않은 이미지는 '정상/탐지 불가'로 분류됩니다.

        사용 모델: {MODELS[MODEL_NAME]['display_name']}

        OOD 감지 방법:
        - confidence: 최대 확률이 {CONFIDENCE_THRESHOLD} 미만인 경우 OOD로 판단 (높을수록 정상)
        - entropy: 확률 분포의 엔트로피가 {ENTROPY_THRESHOLD} 초과인 경우 OOD로 판단 (낮을수록 정상)
    """,
    examples=[
        # 샘플 이미지 예제
        [os.path.join("samples", "정상_1.png"), "confidence"],
        [os.path.join("samples", "정상_2.png"), "confidence"],
        [os.path.join("samples", "오타공.png"), "confidence"],
        [os.path.join("samples", "울음.png"), "confidence"],
        [os.path.join("samples", "피스.png"), "entropy"],
        [os.path.join("samples", "훼손.png"), "entropy"],
    ],
    allow_flagging="never"
)

# 메인 실행
if __name__ == "__main__":
    # 샘플 이미지 디렉토리 확인 및 생성
    sample_dir = "samples"
    ensure_dir(sample_dir)
    print(f"샘플 디렉토리: {os.path.abspath(sample_dir)}")

    # Gradio 실행
    demo.launch(share=True)