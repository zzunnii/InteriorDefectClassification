import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from tqdm.auto import tqdm

from utils.dataset import mixup_data


def cross_entropy_with_soft_labels(outputs, targets):
    """소프트 레이블을 위한 교차 엔트로피 손실 함수"""
    return -torch.sum(targets * torch.log_softmax(outputs, dim=1), dim=1).mean()


def train_one_epoch(model, loader, optimizer, device, mixup_alpha=0.3, use_soft_labels=False,
                    class_weights_tensor=None):
    """한 에폭 학습 (소프트 레이블 + Mixup 지원)"""
    model.train()
    running_loss, running_correct = 0.0, 0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 소프트 레이블 사용 여부에 따라 처리
        if use_soft_labels:
            # Mixup 적용 (50% 확률)
            if np.random.random() < 0.5 and mixup_alpha > 0:
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, mixup_alpha)
                outputs = model(imgs)
                loss = lam * cross_entropy_with_soft_labels(outputs, labels_a) + \
                       (1 - lam) * cross_entropy_with_soft_labels(outputs, labels_b)

                # Mixup 사용 시 정확도 계산 (소프트 레이블 대응)
                preds = outputs.argmax(1)
                targets_a = labels_a.argmax(1)
                targets_b = labels_b.argmax(1)
                correct = (lam * (preds == targets_a).float() +
                           (1 - lam) * (preds == targets_b).float()).sum().item()
            else:
                outputs = model(imgs)
                loss = cross_entropy_with_soft_labels(outputs, labels)

                # 정확도 계산 (소프트 레이블 대응)
                preds = outputs.argmax(1)
                targets = labels.argmax(1)
                correct = (preds == targets).sum().item()

        else:
            # 하드 레이블 사용 시
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

            # Mixup 적용 (50% 확률)
            if np.random.random() < 0.5 and mixup_alpha > 0:
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, mixup_alpha)
                outputs = model(imgs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

                # Mixup 사용 시 정확도 계산
                preds = outputs.argmax(1)
                correct = (lam * (preds == labels_a).float() +
                           (1 - lam) * (preds == labels_b).float()).sum().item()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                correct = (outputs.argmax(1) == labels).sum().item()

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_correct += correct

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct / len(loader.dataset)

    return epoch_loss, epoch_acc


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    """모델 평가"""
    model.eval()
    running_loss, running_correct = 0.0, 0
    preds_all, labels_all = [], []

    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(1)
        running_loss += loss.item() * imgs.size(0)
        running_correct += (preds == labels).sum().item()

        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct / len(loader.dataset)

    # F1 점수 계산
    f1 = f1_score(labels_all, preds_all, average="weighted")

    return epoch_loss, epoch_acc, f1, preds_all, labels_all


def train_model(model, train_loader, val_loader, device, config, model_dir,
                use_soft_labels=False, class_weights_tensor=None):
    """모델 학습 함수"""

    # 모델 디렉토리 생성
    os.makedirs(model_dir, exist_ok=True)

    # 손실 함수 (소프트 레이블 사용 여부에 따라 다름)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 최적화 설정
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 학습 기록
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }

    # 얼리 스탑 카운터 및 최고 성능
    early_stop_counter = 0
    best_val_f1 = 0
    model_path = os.path.join(model_dir, "best_model.pth")

    # 학습 시작
    start_time = time.time()

    for epoch in range(1, config['max_epochs'] + 1):
        # 훈련
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            mixup_alpha=config['mixup_alpha'],
            use_soft_labels=use_soft_labels,
            class_weights_tensor=class_weights_tensor
        )

        # 검증
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        # 스케줄러 업데이트
        scheduler.step(val_loss)

        # 기록 저장
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # 현재 결과 출력
        print(f"[{epoch:3d}/{config['max_epochs']}] "
              f"Train L:{train_loss:.4f} A:{train_acc:.3f} | "
              f"Val L:{val_loss:.4f} A:{val_acc:.3f} F1:{val_f1:.3f}")

        # 최고 F1 스코어 모델 저장
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            early_stop_counter = 0
            print(f"🔥 New best model saved! (F1: {val_f1:.4f})")
        else:
            early_stop_counter += 1

        # 얼리 스탑
        if early_stop_counter >= config['patience']:
            print(f"⛔ Early stopping at epoch {epoch}")
            break

    # 학습 완료
    total_time = time.time() - start_time
    print(f"총 학습 시간: {total_time / 60:.2f}분")

    # 최고 모델 로드
    model.load_state_dict(torch.load(model_path))

    return model, history, best_val_f1, total_time


def test_model(model, test_loader, device, class_names, class_weights_tensor=None):
    """모델 테스트 및 성능 평가"""
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"테스트 결과: Acc:{test_acc:.3f} F1:{test_f1:.3f}")

    # 혼동 행렬 생성
    cm = confusion_matrix(test_labels, test_preds)

    # 클래스별 정확도 계산
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        if np.sum(cm[i]) > 0:
            class_accuracy[class_name] = cm[i, i] / np.sum(cm[i])
        else:
            class_accuracy[class_name] = 0.0

    return {
        "test_acc": test_acc,
        "test_f1": test_f1,
        "confusion_matrix": cm,
        "class_accuracy": class_accuracy,
        "test_preds": test_preds,
        "test_labels": test_labels
    }