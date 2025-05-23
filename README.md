인테리어 결함 분류 프로젝트

## 프로젝트 개요

이 프로젝트는 인테리어 결함을 자동으로 분류하기 위한 딥러닝 기반 시스템입니다. 여러 모델 아키텍처(ResNet34, EfficientNet B0, DenseNet121, ConvNext Base)를 활용하여 학습 및 평가를 수행하며, 데이터 불균형 문제를 해결하기 위해 소프트 레이블링, Mixup 증강, 클래스 가중치 조정 등의 기법을 사용합니다. 또한, AdaBoost 스타일의 부스팅 앙상블과 단순 평균 앙상블을 통해 예측 성능을 향상시켰습니다. Gradio를 사용한 인터랙티브 데모는 Out-of-Distribution (OOD) 감지 기능을 포함하여 실제 환경에서의 활용성을 높입니다. 학습 결과는 혼동 행렬, 학습 곡선, 클래스별 정확도 등 다양한 시각화 도구를 통해 분석됩니다.

## 데이터 출처

이 프로젝트에서 사용된 데이터셋은 Dacon 인테리어 결함 분류 대회에서 제공됩니다. 데이터 다운로드는 위 링크를 통해 가능하며, 데이터에 대한 모든 라이선스와 저작권은 Dacon에 있습니다.

## 프로젝트 구조

프로젝트는 다음과 같은 주요 파일과 디렉토리로 구성되어 있습니다:

- `dataset.py`: 커스텀 데이터셋 클래스(`InteriorDefectDataset`)와 데이터 로더를 정의하며, 소프트 레이블과 Mixup 증강을 지원합니다.
- `utils.py`: 재현성을 위한 시드 고정, 이미지 경로 수집, 클래스 가중치 계산, 소프트 레이블 매트릭스 생성 등의 유틸리티 함수를 제공합니다.
- `models.py`: ResNet34, EfficientNet B0, DenseNet121, ConvNext Base 모델을 생성하는 함수를 포함합니다.
- `train_utils.py`: 모델 학습, 평가, 테스트를 위한 함수를 정의하며, 소프트 레이블과 클래스 가중치를 지원합니다.
- `visualize.py`: 혼동 행렬, 학습 곡선, 클래스별 정확도, 모델별 성능 비교를 시각화하는 함수를 제공합니다.
- `config.py`: 데이터 경로, 모델 설정, 학습 하이퍼파라미터, 소프트 레이블 매핑 등을 정의합니다.
- `main.py`: 단일 모델 학습 및 평가를 수행하는 메인 스크립트입니다.
- `ensemble.py`: 부스팅 앙상블과 단순 평균 앙상블을 구현하며, 모델별 성능 비교를 제공합니다.
- `gradio_demo.py`: Gradio를 사용한 인터랙티브 데모로, OOD 감지(신뢰도 및 엔트로피 기반)를 지원합니다.
- `post_augmentation.py`: 학습 데이터에 오프라인 데이터 증강(Random Cutout, RSNA 블랙아웃)을 적용하고 train/val/test로 분할합니다.
- `requirements.txt`: 프로젝트 실행에 필요한 Python 패키지 목록입니다.

## 설치 방법

1. **PyTorch 설치**: PyTorch와 관련 패키지를 설치하려면 다음 명령어를 실행하세요 (CUDA 11.8 기준):

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```


2. **필수 패키지 설치**: 프로젝트 실행에 필요한 Python 패키지를 설치합니다. 다음 명령어를 사용하세요:

   ```bash
   pip install -r requirements.txt
   ```

3. **데이터셋 준비**:

   - Dacon 인테리어 결함 분류 대회에서 데이터를 다운로드하세요.

   - 데이터를 `config.py`의 `DATA_DIR`에 지정된 경로에 저장하거나 아래 방법에 따라 실행하세요.

   - `post_augmentation.py`를 실행하여 데이터를 train/val/test로 분할하고 학습 데이터에 증강을 적용하세요:

     ```bash
     python post_augmentation.py --source_root <원본 데이터 경로> --dest_root <증강 데이터 저장 경로>
     ```

## 사용 방법

### 1. 모델 학습

`main.py`를 실행하여 단일 모델 또는 모든 모델을 학습합니다:
```bash
python main.py --model all --data-dir <데이터 경로> --use-soft-labels
```

- `--model`: 학습할 모델 이름(`resnet34`, `efficientnet_b0`, `densenet121`, `convnext_base`, 또는 `all`)을 지정합니다.
- `--data-dir`: 데이터셋 디렉토리 경로를 지정합니다.
- `--use-soft-labels`: 소프트 레이블링을 활성화합니다.

학습 결과는 `results/<모델명>` 디렉토리에 저장되며, 혼동 행렬, 학습 곡선, 클래스별 정확도 시각화 이미지와 JSON 결과 파일이 포함됩니다.

### 2. 앙상블 평가

`ensemble.py`를 실행하여 부스팅 및 단순 평균 앙상블을 평가합니다:

```bash
python ensemble.py --data-dir <데이터 경로> --optimize-weights
```

- `--optimize-weights`: AdaBoost 스타일의 부스팅 가중치를 계산합니다.
- 결과는 `results/ensemble` 디렉토리에 저장됩니다.

### 3. Gradio 데모 실행

`gradio_demo.py`를 실행하여 인터랙티브 웹 인터페이스를 통해 모델 예측을 확인합니다:

```bash
python gradio_demo.py
```

- Gradio 인터페이스는 이미지 업로드를 통해 결함 분류를 수행하며, OOD 감지(신뢰도 또는 엔트로피 기반)를 지원합니다.
- 샘플 이미지는 `samples` 디렉토리에 저장됩니다.
- 웹 브라우저에서 인터페이스에 접근하여 결과를 확인할 수 있습니다.

## 결과 및 시각화

- **혼동 행렬**: 테스트 데이터에 대한 예측 성능을 시각화합니다. 아래는 ConvNext Base 모델의 혼동 행렬 예시입니다:

- **학습 곡선**: 학습 및 검증 데이터의 손실, 정확도, F1 스코어를 그래프로 표시합니다.

- **클래스별 정확도**: 각 클래스의 예측 정확도를 바 차트로 시각화합니다.

- **모델별 성능 비교**: 각 모델과 앙상블의 정확도 및 F1 스코어를 비교합니다.

- **Gradio 데모**: 예측 결과, 클래스별 확률 바 차트, OOD 판단 기준 게이지를 제공합니다.

## 요구사항

- Python 3.8 이상
- PyTorch (CUDA 지원 권장)
- 기타 필수 패키지는 `requirements.txt`에 명시되어 있습니다.

## 라이선스

이 프로젝트에서 사용된 데이터셋의 모든 라이선스와 저작권은 Dacon에 있습니다. 코드와 관련된 라이선스는 별도로 명시되지 않았으므로, 사용 시 Dacon의 데이터 사용 정책을 준수해야 합니다.

## 참고 사항

- 데이터 경로(`DATA_DIR`)는 `config.py`에서 적절히 설정해야 합니다.
- Windows 환경에서는 `num_workers=0`으로 설정하여 데이터 로딩 문제를 방지할 수 있습니다.
- OOD 감지 임계값(`CONFIDENCE_THRESHOLD`, `ENTROPY_THRESHOLD`)은 `gradio_demo.py`에서 조정 가능합니다.
- 부스팅 앙상블은 AdaBoost 기반 가중치를 사용하여 모델별 예측을 결합합니다.

## 문의

추가 질문이나 문제가 있을 경우, Dacon 포럼 또는 프로젝트 관련 문의 채널을 통해 연락 바랍니다.