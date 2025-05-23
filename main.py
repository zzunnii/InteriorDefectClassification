#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
인테리어 결함 분류 모델 학습 메인 스크립트
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

import config
from utils import (
    seed_everything, collect_images, calculate_class_weights,
    create_soft_label_matrix, ensure_dir, save_json,
    get_model_by_name, create_data_loaders,
    train_model, test_model,
    plot_confusion_matrix, plot_learning_curves, plot_class_accuracy
)


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="인테리어 결함 분류 모델 학습")

    parser.add_argument("--model", type=str, default="all",
                        choices=list(config.MODELS.keys()) + ["all"],
                        help="학습할 모델 이름 (기본값: all)")

    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR,
                        help=f"데이터 디렉토리 경로 (기본값: {config.DATA_DIR})")

    parser.add_argument("--use-soft-labels", action="store_true", default=True,
                        help="소프트 레이블 사용 여부 (기본값: False)")

    parser.add_argument("--seed", type=int, default=42,
                        help="난수 시드 (기본값: 42)")

    parser.add_argument("--no-cuda", action="store_true",
                        help="CUDA 사용하지 않음 (기본값: False)")

    return parser.parse_args()


def prepare_data(data_dir, class_to_idx):
    """데이터 준비"""
    # 경로 설정
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # 이미지 경로 수집
    train_paths, train_labels = collect_images(train_dir, class_to_idx)
    val_paths, val_labels = collect_images(val_dir, class_to_idx)
    test_paths, test_labels = collect_images(test_dir, class_to_idx)

    # 데이터 형식 맞추기
    train_data = [{"image": path, "label": label} for path, label in zip(train_paths, train_labels)]
    val_data = [{"image": path, "label": label} for path, label in zip(val_paths, val_labels)]
    test_data = [{"image": path, "label": label} for path, label in zip(test_paths, test_labels)]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data, train_labels


def train_single_model(model_name, train_data, val_data, test_data, class_names,
                       class_to_idx, train_labels, device, use_soft_labels=False):
    """단일 모델 학습 및 평가"""
    # 모델 설정 가져오기
    model_config = config.MODELS[model_name]
    model_dir = model_config["result_dir"]
    ensure_dir(model_dir)

    print(f"\n\n{'=' * 20}  {model_config['display_name']} 학습 시작  {'=' * 20}")

    # 클래스 가중치 계산
    num_classes = len(class_names)
    class_weights = calculate_class_weights(train_labels, num_classes)
    class_weights_tensor = torch.tensor(class_weights).float().to(device)
    print("클래스 가중치 ➜", np.round(class_weights, 3))

    # 소프트 레이블 매트릭스 생성 (사용 설정 시)
    soft_label_matrix = None
    if use_soft_labels:
        soft_label_matrix = create_soft_label_matrix(
            class_names, config.SOFT_LABEL_MAPPING, default_weight=config.SOFT_LABEL_WEIGHT
        )
        print(f"소프트 레이블 적용 (기본 가중치: {config.SOFT_LABEL_WEIGHT})")

    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        batch_size=config.TRAIN_CONFIG["batch_size"],
        soft_label_matrix=soft_label_matrix,
        img_size=config.TRAIN_CONFIG["image_size"],
        num_workers=config.TRAIN_CONFIG["num_workers"]
    )

    # 모델 생성
    model = get_model_by_name(model_name, num_classes).to(device)

    # 모델 학습
    model, history, best_val_f1, training_time = train_model(
        model, train_loader, val_loader, device,
        config=config.TRAIN_CONFIG,
        model_dir=model_dir,
        use_soft_labels=use_soft_labels,
        class_weights_tensor=class_weights_tensor
    )

    # 테스트 평가
    test_results = test_model(
        model, test_loader, device, class_names, class_weights_tensor
    )

    # 결과 시각화 및 저장
    plot_confusion_matrix(
        test_results["confusion_matrix"], class_names,
        title=f"Confusion Matrix - {model_config['display_name']}",
        save_path=os.path.join(model_dir, "confusion_matrix.png")
    )

    plot_learning_curves(
        history, save_path=os.path.join(model_dir, "learning_curves.png")
    )

    plot_class_accuracy(
        test_results["class_accuracy"],
        title=f"Class-wise Accuracy - {model_config['display_name']}",
        save_path=os.path.join(model_dir, "class_accuracy.png")
    )

    # 결과 정보 저장
    results_info = {
        "model_name": model_name,
        "display_name": model_config["display_name"],
        "use_soft_labels": use_soft_labels,
        "training_config": config.TRAIN_CONFIG,
        "class_weights": class_weights,
        "best_val_f1": float(best_val_f1),
        "training_time": float(training_time),
        "training_time_min": float(training_time) / 60,
        "test_accuracy": float(test_results["test_acc"]),
        "test_f1": float(test_results["test_f1"]),
        "class_accuracy": test_results["class_accuracy"],
        "history": {
            "train_loss": [float(x) for x in history["train_loss"]],
            "train_acc": [float(x) for x in history["train_acc"]],
            "val_loss": [float(x) for x in history["val_loss"]],
            "val_acc": [float(x) for x in history["val_acc"]],
            "val_f1": [float(x) for x in history["val_f1"]]
        },
        "epochs_trained": len(history["train_loss"])
    }

    # 결과 JSON 저장
    save_json(results_info, os.path.join(model_dir, "results.json"))

    # 메타데이터 저장
    metadata = {
        "model_name": model_name,
        "display_name": model_config["display_name"],
        "num_classes": num_classes,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "use_soft_labels": use_soft_labels,
        "soft_label_mapping": config.SOFT_LABEL_MAPPING if use_soft_labels else None,
        "soft_label_weight": config.SOFT_LABEL_WEIGHT if use_soft_labels else None
    }

    save_json(metadata, os.path.join(model_dir, "metadata.json"))

    # 메모리 절약을 위해 모델과 관련 객체 삭제
    del model
    torch.cuda.empty_cache()

    return results_info


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

    # 결과 디렉토리 생성
    ensure_dir(config.RESULTS_DIR)

    # 클래스 정보 로드
    train_dir = os.path.join(args.data_dir, "train")
    class_names = sorted(os.listdir(train_dir))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    num_classes = len(class_names)

    print(f"클래스 수: {num_classes}")
    print(f"클래스 이름: {class_names}")

    # 데이터 준비
    train_data, val_data, test_data, train_labels = prepare_data(args.data_dir, class_to_idx)

    # 학습할 모델 결정
    models_to_train = list(config.MODELS.keys()) if args.model == "all" else [args.model]

    # 결과 저장
    all_results = {}

    # 각 모델 학습 및 평가
    for model_name in models_to_train:
        try:
            results_info = train_single_model(
                model_name, train_data, val_data, test_data,
                class_names, class_to_idx, train_labels,
                device, use_soft_labels=args.use_soft_labels
            )
            all_results[model_name] = results_info
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # 전체 결과 저장
    result_summary = {
        "models": all_results,
        "args": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 소프트 레이블 사용 여부에 따라 다른 파일명 사용
    summary_filename = "results_summary_soft.json" if args.use_soft_labels else "results_summary.json"
    save_json(result_summary, os.path.join(config.RESULTS_DIR, summary_filename))

    # 모델별 성능 비교 표 출력
    print("\n\n==== 모델별 테스트 결과 비교 ====")
    results_df = pd.DataFrame({
        "모델": [config.MODELS[m]["display_name"] for m in all_results.keys()],
        "정확도": [all_results[m]["test_accuracy"] for m in all_results.keys()],
        "F1 스코어": [all_results[m]["test_f1"] for m in all_results.keys()],
        "학습 시간(분)": [all_results[m]["training_time_min"] for m in all_results.keys()]
    })

    print(results_df.sort_values("F1 스코어", ascending=False).to_string(index=False))

    # 최고 성능 모델 출력
    if all_results:
        best_model = max(all_results.items(), key=lambda x: x[1]["test_f1"])
        print(f"\n🏆 최고 성능 모델: {config.MODELS[best_model[0]]['display_name']} (F1: {best_model[1]['test_f1']:.4f})")

    print("\n모든 모델 학습 완료!")


if __name__ == "__main__":
    main()