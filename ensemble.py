#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
인테리어 결함 분류 모델 부스팅 앙상블 평가
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import config
from utils import (
    seed_everything, collect_images, ensure_dir, save_json, load_json,
    get_model_by_name, get_transforms,
    plot_confusion_matrix, plot_class_accuracy, plot_models_comparison
)


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="인테리어 결함 분류 모델 부스팅 앙상블 평가")

    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR,
                        help=f"데이터 디렉토리 경로 (기본값: {config.DATA_DIR})")

    parser.add_argument("--seed", type=int, default=42,
                        help="난수 시드 (기본값: 42)")

    parser.add_argument("--optimize-weights", action="store_true", default=True,
                        help="부스팅 가중치 계산 (기본값: True)")

    parser.add_argument("--no-cuda", action="store_true",
                        help="CUDA 사용하지 않음 (기본값: False)")

    return parser.parse_args()


class InteriorDataset(torch.utils.data.Dataset):
    """분류를 위한 테스트 데이터셋"""

    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx], self.labels[idx]
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"이미지 로딩 에러 ({path}): {e}")
            img = Image.new("RGB", (256, 256), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


@torch.inference_mode()
def predict_with_model(model, data_loader, device):
    """모델 예측 수행"""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for imgs, labels in data_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.append(probs.cpu())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    return torch.cat(all_probs, dim=0), all_preds, all_labels


def load_models(model_names, num_classes, device):
    """모델들 로드"""
    models = {}
    for model_name in model_names:
        try:
            model_config = config.MODELS[model_name]
            model_path = os.path.join(model_config["result_dir"], "best_model.pth")

            if not os.path.exists(model_path):
                print(f"모델 파일을 찾을 수 없습니다: {model_path}")
                continue

            model = get_model_by_name(model_name, num_classes).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            models[model_name] = model
            print(f"모델 로드 성공: {model_config['display_name']}")
        except Exception as e:
            print(f"모델 로드 실패 ({model_name}): {e}")

    return models


def calculate_boosting_weights(probs_dict, true_labels, class_names, epsilon=1e-10):
    """AdaBoost 스타일의 부스팅 가중치 계산"""
    num_samples = len(true_labels)
    model_names = list(probs_dict.keys())
    num_models = len(model_names)
    num_classes = len(class_names)

    # 초기 샘플 가중치 (균등 분포)
    sample_weights = np.ones(num_samples) / num_samples

    # 각 모델의 부스팅 가중치
    model_weights = {}

    # 모델 순서 정렬 (성능 기준)
    model_errors = {}
    for model_name in model_names:
        probs = probs_dict[model_name]
        preds = probs.argmax(dim=1).numpy()
        # 오류율 계산 (가중치 적용)
        incorrect = (preds != np.array(true_labels))
        error_rate = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
        model_errors[model_name] = error_rate

    # 오류율 기준으로 모델 정렬 (오름차순)
    sorted_models = sorted(model_errors.items(), key=lambda x: x[1])

    # 부스팅 가중치 계산
    for model_idx, (model_name, error_rate) in enumerate(sorted_models):
        # 오류율이 0인 경우 처리 (완벽한 모델)
        error_rate = max(error_rate, epsilon)
        error_rate = min(error_rate, 1.0 - epsilon)

        # AdaBoost 가중치 계산식
        alpha = 0.5 * np.log((1 - error_rate) / error_rate)
        model_weights[model_name] = alpha

        # 다음 모델을 위한 샘플 가중치 업데이트
        probs = probs_dict[model_name]
        preds = probs.argmax(dim=1).numpy()
        incorrect = (preds != np.array(true_labels))

        # 샘플 가중치 업데이트 (AdaBoost 방식)
        sample_weights = sample_weights * np.exp(alpha * incorrect)
        sample_weights = sample_weights / np.sum(sample_weights)

    # 모델 가중치 정규화 (합이 1이 되도록)
    total_weight = sum(model_weights.values())
    if total_weight > 0:
        for model_name in model_weights:
            model_weights[model_name] /= total_weight
    else:
        # 모든 가중치가 0인 경우 균등 분배
        for model_name in model_weights:
            model_weights[model_name] = 1.0 / len(model_weights)

    print("\n부스팅 모델 가중치:")
    for model_name, weight in sorted(model_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {config.MODELS[model_name]['display_name']}: {weight:.4f}")

    return {"boosting": model_weights}


def boosting_ensemble_predict(probs_dict, boosting_weights):
    """부스팅 앙상블 예측 수행"""
    model_names = list(probs_dict.keys())
    num_samples = probs_dict[model_names[0]].shape[0]
    num_classes = probs_dict[model_names[0]].shape[1]

    # 가중치 앙상블 예측
    weighted_probs = torch.zeros((num_samples, num_classes))

    # 모델별 가중치 적용
    model_weights = boosting_weights["boosting"]
    for model_name, weight in model_weights.items():
        weighted_probs += weight * probs_dict[model_name]

    # 예측값 계산
    weighted_preds = weighted_probs.argmax(dim=1).tolist()

    return weighted_probs, weighted_preds


def simple_ensemble_predict(probs_dict):
    """단순 평균 앙상블 예측"""
    all_probs = torch.stack(list(probs_dict.values()))
    mean_probs = all_probs.mean(dim=0)
    preds = mean_probs.argmax(dim=1).tolist()

    return mean_probs, preds


def evaluate_models(models_dict, test_loader, device, class_names):
    """모든 모델과 앙상블 평가"""
    results = {}
    probs_dict = {}
    true_labels = None

    # 개별 모델 평가
    for model_name, model in models_dict.items():
        probs, preds, labels = predict_with_model(model, test_loader, device)

        if true_labels is None:
            true_labels = labels

        acc = (np.array(preds) == np.array(labels)).mean()
        f1 = f1_score(labels, preds, average="weighted")

        display_name = config.MODELS[model_name]["display_name"]
        print(f"{display_name}: Acc={acc:.3f}, F1={f1:.3f}")

        probs_dict[model_name] = probs

        results[model_name] = {
            "display_name": display_name,
            "accuracy": acc,
            "f1_score": f1,
            "predictions": preds
        }

    return results, probs_dict, true_labels


def main():
    """메인 함수"""
    # 명령줄 인자 파싱
    args = parse_args()

    # 시드 고정
    seed_everything(args.seed)

    # CUDA 설정
    use_cuda = config.USE_CUDA and torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda and config.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

    print(f"Device: {device}")

    # 앙상블 디렉토리 생성
    ensure_dir(config.ENSEMBLE_DIR)

    # 클래스 정보 로드
    train_dir = os.path.join(args.data_dir, "train")
    class_names = sorted(os.listdir(train_dir))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    num_classes = len(class_names)

    print(f"클래스 수: {num_classes}")

    # 테스트 데이터 준비
    test_dir = os.path.join(args.data_dir, "test")
    test_paths, test_labels = collect_images(test_dir, class_to_idx)
    print(f"테스트 데이터 수: {len(test_paths)}")

    # 데이터 로더 생성
    _, test_transform = get_transforms(img_size=config.TRAIN_CONFIG["image_size"])
    test_dataset = InteriorDataset(test_paths, test_labels, test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TRAIN_CONFIG["batch_size"],
        shuffle=False, num_workers=0
    )

    # 모델 로드
    models = load_models(config.ENSEMBLE_MODELS, num_classes, device)

    if len(models) < 2:
        print("앙상블을 위한 모델이 2개 이상 필요합니다.")
        return

    # 모델별 성능 평가
    results, probs_dict, true_labels = evaluate_models(models, test_loader, device, class_names)

    # 부스팅 가중치 계산 또는 로드
    boosting_weights_path = os.path.join(config.ENSEMBLE_DIR, "boosting_weights.json")

    if args.optimize_weights:
        print("\n부스팅 가중치 계산 중...")
        boosting_weights = calculate_boosting_weights(probs_dict, true_labels, class_names)
        save_json(boosting_weights, boosting_weights_path)
        print(f"부스팅 가중치 저장: {boosting_weights_path}")
    elif os.path.exists(boosting_weights_path):
        print(f"기존 부스팅 가중치 로드: {boosting_weights_path}")
        boosting_weights = load_json(boosting_weights_path)
    else:
        print("부스팅 가중치 파일이 없어 균등 가중치를 사용합니다.")
        boosting_weights = {"boosting": {model: 1.0 / len(models) for model in models}}

    # 부스팅 앙상블 예측
    boosting_probs, boosting_preds = boosting_ensemble_predict(
        probs_dict, boosting_weights
    )

    boosting_acc = (np.array(boosting_preds) == np.array(true_labels)).mean()
    boosting_f1 = f1_score(true_labels, boosting_preds, average="weighted")

    print(f"\n🔹 부스팅 앙상블: Acc={boosting_acc:.3f}, F1={boosting_f1:.3f}")

    # 단순 평균 앙상블 예측 (비교용)
    simple_probs, simple_preds = simple_ensemble_predict(probs_dict)

    simple_acc = (np.array(simple_preds) == np.array(true_labels)).mean()
    simple_f1 = f1_score(true_labels, simple_preds, average="weighted")

    print(f"🔹 단순 평균 앙상블: Acc={simple_acc:.3f}, F1={simple_f1:.3f}")

    # 앙상블 결과 추가
    results["boosting_ensemble"] = {
        "display_name": "부스팅 앙상블",
        "accuracy": boosting_acc,
        "f1_score": boosting_f1,
        "predictions": boosting_preds
    }

    results["simple_ensemble"] = {
        "display_name": "단순 평균 앙상블",
        "accuracy": simple_acc,
        "f1_score": simple_f1,
        "predictions": simple_preds
    }

    # 결과 시각화 및 저장
    # 혼동 행렬
    cm = confusion_matrix(true_labels, boosting_preds)
    plot_confusion_matrix(
        cm, class_names,
        title="Confusion Matrix - Boosting Ensemble",
        save_path=os.path.join(config.ENSEMBLE_DIR, "confusion_matrix_boosting.png")
    )

    # 모델 성능 비교
    plot_models_comparison(
        {k: {"test_acc": v["accuracy"]} for k, v in results.items()},
        metric="accuracy",
        save_path=os.path.join(config.ENSEMBLE_DIR, "accuracy_comparison_boosting.png")
    )

    plot_models_comparison(
        {k: {"test_f1": v["f1_score"]} for k, v in results.items()},
        metric="f1",
        save_path=os.path.join(config.ENSEMBLE_DIR, "f1_comparison_boosting.png")
    )

    # 클래스별 정확도 계산
    class_acc = defaultdict(float)
    for true_label, pred_label in zip(true_labels, boosting_preds):
        if true_label == pred_label:
            class_name = class_names[true_label]
            class_acc[class_name] += 1

    for class_name in class_names:
        class_count = sum(1 for label in true_labels if label == class_to_idx[class_name])
        if class_count > 0:
            class_acc[class_name] /= class_count

    # 클래스별 정확도 시각화
    plot_class_accuracy(
        class_acc,
        title="Class-wise Accuracy - Boosting Ensemble",
        save_path=os.path.join(config.ENSEMBLE_DIR, "class_accuracy_boosting.png")
    )

    # 클래스별 분류 보고서
    classification_rep = classification_report(
        true_labels, boosting_preds,
        target_names=class_names,
        digits=3,
        output_dict=True
    )

    # 모든 결과 저장
    final_results = {
        "models": results,
        "boosting_weights": boosting_weights,
        "classification_report": classification_rep,
        "class_accuracy": class_acc
    }

    save_json(final_results, os.path.join(config.ENSEMBLE_DIR, "boosting_results.json"))

    # 종합 결과 출력
    print("\n==== 모델별 성능 비교 ====")
    comparison_df = pd.DataFrame({
        "모델": [results[m]["display_name"] for m in results.keys()],
        "정확도": [results[m]["accuracy"] for m in results.keys()],
        "F1 스코어": [results[m]["f1_score"] for m in results.keys()]
    })

    print(comparison_df.sort_values("F1 스코어", ascending=False).to_string(index=False))

    # 성능 향상 정도
    best_single_f1 = max([v["f1_score"] for k, v in results.items()
                          if k not in ["boosting_ensemble", "simple_ensemble"]])

    improvement = boosting_f1 - best_single_f1
    print(f"\n🔹 최고 단일 모델 대비 성능 향상: {improvement:.3f} ({improvement * 100:.1f}%)")

    print("\n부스팅 앙상블 평가 완료!")


if __name__ == "__main__":
    from PIL import Image

    main()