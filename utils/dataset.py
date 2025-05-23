import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class InteriorDefectDataset(Dataset):
    """인테리어 결함 분류를 위한 커스텀 데이터셋, 소프트 레이블 지원"""

    def __init__(self, data, transform=None, soft_label_matrix=None, train_mode=False):
        """
        Args:
            data: 데이터 딕셔너리 리스트 [{"image": path, "label": label}, ...]
            transform: 이미지 변환 파이프라인
            soft_label_matrix: 소프트 레이블 매트릭스
            train_mode: 학습 모드 여부
        """
        self.data = data
        self.transform = transform
        self.soft_label_matrix = soft_label_matrix
        self.train_mode = train_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]["image"]
        label = self.data[idx]["label"]

        # 한글 경로 지원을 위한 이미지 로딩
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"이미지 로딩 에러 ({path}): {e}")
            img = Image.new("RGB", (256, 256), (0, 0, 0))  # 에러 시 검은색 더미 이미지

        if self.transform:
            img = self.transform(img)

        # 소프트 레이블 적용 (학습 모드이고 소프트 레이블 매트릭스가 제공된 경우)
        if self.train_mode and self.soft_label_matrix is not None:
            soft_label = self.soft_label_matrix[label]
            return img, torch.tensor(soft_label, dtype=torch.float32)
        else:
            # 테스트/검증 모드이거나 소프트 레이블이 없는 경우 일반 레이블 반환
            return img, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size=256):
    """학습 및 검증/테스트용 이미지 변환 파이프라인 생성"""

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def mixup_data(x, y, alpha=0.3):
    """Mixup 데이터 증강 함수
    소프트 레이블과 하드 레이블 모두 지원
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def create_data_loaders(train_data, val_data, test_data, batch_size,
                        soft_label_matrix=None, img_size=256, num_workers=4):
    """데이터셋과 데이터로더를 생성하는 유틸리티 함수"""

    train_transform, val_transform = get_transforms(img_size)

    # 데이터셋 생성
    train_dataset = InteriorDefectDataset(
        train_data, transform=train_transform,
        soft_label_matrix=soft_label_matrix, train_mode=True
    )

    val_dataset = InteriorDefectDataset(
        val_data, transform=val_transform
    )

    test_dataset = InteriorDefectDataset(
        test_data, transform=val_transform
    )

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, test_loader