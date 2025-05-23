import os
import random
import shutil
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path
import numpy as np
import argparse

def create_train_val_test_split(source_root, dest_root, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    데이터셋을 train, val, test로 분할하고 train 세트에 오프라인 증강 적용
    """
    # 시드 고정
    random.seed(seed)
    np.random.seed(seed)

    # 결과 디렉토리 생성
    train_dir = os.path.join(dest_root, 'train')
    val_dir = os.path.join(dest_root, 'val')
    test_dir = os.path.join(dest_root, 'test')

    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    # 루트 디렉토리에서 모든 클래스 폴더 찾기
    class_folders = [f for f in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, f))]
    print(f"찾은 클래스 수: {len(class_folders)}")

    # 각 클래스별로 처리
    for class_name in tqdm(class_folders, desc="클래스 처리 중"):
        # 클래스별 출력 디렉토리 생성
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)

        for directory in [class_train_dir, class_val_dir, class_test_dir]:
            os.makedirs(directory, exist_ok=True)

        # 클래스 폴더 내 모든 이미지 파일 찾기
        class_path = os.path.join(source_root, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # 이미지 파일 섞기 (시드 고정됨)
        random.shuffle(image_files)

        # 분할 인덱스 계산
        n_total = len(image_files)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]

        # 학습 데이터 복사 및 증강
        for img_file in tqdm(train_files, desc=f"{class_name} 학습 데이터 처리 중", leave=False):
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(class_train_dir, img_file)

            # 원본 이미지 복사
            shutil.copy2(src_path, dst_path)

            # 한글 경로 이미지 읽기 - numpy로 파일 읽은 후 디코딩
            try:
                with open(src_path, 'rb') as f:
                    file_bytes = np.frombuffer(f.read(), np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if image is None:
                    print(f"경고: {src_path} 파일을 읽을 수 없습니다.")
                    continue

                base_name, ext = os.path.splitext(img_file)

                # 1. New Random Cutout 적용
                cutout_image = apply_random_cutout(image.copy())
                cutout_path = os.path.join(class_train_dir, f"{base_name}_cutout{ext}")
                # 한글 경로에 이미지 저장
                cv2.imencode(ext, cutout_image)[1].tofile(cutout_path)

                # 2. RSNA 블랙아웃 증강 적용
                rsna_image = apply_rsna_blackout(image.copy())
                rsna_path = os.path.join(class_train_dir, f"{base_name}_rsna{ext}")
                # 한글 경로에 이미지 저장
                cv2.imencode(ext, rsna_image)[1].tofile(rsna_path)

            except Exception as e:
                print(f"오류 발생: {src_path} 파일 처리 중 - {str(e)}")

        # 검증 데이터 복사 (증강 없음)
        for img_file in val_files:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(class_val_dir, img_file)
            shutil.copy2(src_path, dst_path)

        # 테스트 데이터 복사 (증강 없음)
        for img_file in test_files:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(class_test_dir, img_file)
            shutil.copy2(src_path, dst_path)

    print("데이터셋 분할 및 증강 완료!")


def apply_random_cutout(image):
    """New Random Cutout 적용 (이미지에서 설명된 방식으로)"""
    h, w = image.shape[:2]

    # 랜덤으로 8-40 픽셀 크기 결정
    cutout_size = random.randint(8, 40)

    # 6가지 패턴 중 하나 선택
    pattern = random.randint(1, 6)

    # 패턴에 따라 자르기 적용
    if pattern == 1:  # top_left, bottom_left
        image[:cutout_size, :cutout_size] = 0  # top_left
        image[h - cutout_size:, :cutout_size] = 0  # bottom_left
    elif pattern == 2:  # top_left, top_right
        image[:cutout_size, :cutout_size] = 0  # top_left
        image[:cutout_size, w - cutout_size:] = 0  # top_right
    elif pattern == 3:  # top_right, bottom_right
        image[:cutout_size, w - cutout_size:] = 0  # top_right
        image[h - cutout_size:, w - cutout_size:] = 0  # bottom_right
    elif pattern == 4:  # bottom_left, bottom_right
        image[h - cutout_size:, :cutout_size] = 0  # bottom_left
        image[h - cutout_size:, w - cutout_size:] = 0  # bottom_right
    elif pattern == 5:  # top_left, bottom_left and bottom_left, bottom_right
        image[:cutout_size, :cutout_size] = 0  # top_left
        image[h - cutout_size:, :cutout_size] = 0  # bottom_left
        image[h - cutout_size:, w - cutout_size:] = 0  # bottom_right
    else:  # pattern == 6, top_right, bottom_right and bottom_left, bottom_right
        image[:cutout_size, w - cutout_size:] = 0  # top_right
        image[h - cutout_size:, w - cutout_size:] = 0  # bottom_right
        image[h - cutout_size:, :cutout_size] = 0  # bottom_left

    return image


def apply_rsna_blackout(image):
    """RSNA 스타일 블랙아웃 증강 적용"""
    h, w = image.shape[:2]

    # 블랙아웃할 영역 개수 결정 (1-3개)
    num_regions = random.randint(1, 3)

    for _ in range(num_regions):
        # 블랙아웃 위치와 크기 랜덤 결정
        region_w = random.randint(w // 10, w // 3)
        region_h = random.randint(h // 10, h // 3)

        x = random.randint(0, w - region_w)
        y = random.randint(0, h - region_h)

        # 블랙아웃 적용 (검은색)
        image[y:y + region_h, x:x + region_w] = 0

    return image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="데이터 PostAugementation 스크립트")
    parser.add_argument("--source_root", type=str, required=True, help="원본 데이터 경로")
    parser.add_argument("dest_root", type=str, required=True, help="증강 데이터 저장 경로")

    args = parser.parse_args()

    # 소스 및 대상 경로 설정
    SOURCE_ROOT = args.source_root
    DEST_ROOT = args.dest_root

    os.makedirs(DEST_ROOT, exist_ok=True)
    # 시드 값 설정
    SEED = 42

    # 데이터셋 분할 및 증강 실행
    create_train_val_test_split(
        source_root=SOURCE_ROOT,
        dest_root=DEST_ROOT,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=SEED
    )