import torch
import torch.nn as nn
from torchvision import models
import timm


def create_resnet34(num_classes):
    """ResNet34 모델 생성"""
    model = models.resnet34(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def create_efficientnet_b0(num_classes):
    """EfficientNet B0 모델 생성"""
    model = timm.create_model('efficientnet_b0', pretrained=True)
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    else:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    return model


def create_densenet121(num_classes):
    """DenseNet121 모델 생성"""
    model = models.densenet121(weights="IMAGENET1K_V1")
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


def create_convnext_base(num_classes):
    """ConvNext Base 모델 생성"""
    model = models.convnext_base(weights="IMAGENET1K_V1")
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, num_classes)
    )
    return model


def get_model_by_name(model_name, num_classes):
    """모델 이름으로 모델 생성 함수 호출"""
    model_funcs = {
        'resnet34': create_resnet34,
        'efficientnet_b0': create_efficientnet_b0,
        'densenet121': create_densenet121,
        'convnext_base': create_convnext_base
    }

    if model_name not in model_funcs:
        raise ValueError(f"지원되지 않는 모델: {model_name}")

    return model_funcs[model_name](num_classes)