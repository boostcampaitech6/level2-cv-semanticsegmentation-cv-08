import albumentations as A


def get_transform_function(transform_function_str, config):
    if transform_function_str == "baseTransform":
        return baseTransform(config)


def baseTransform(config):
    return A.Compose(
        [
            A.Resize(config["input_size"], config["input_size"]),
        ]
    )


def medicalTransform(config):
    return A.Compose(
        [
            # 이미지를 임의의 크기로 자르고 다시 조정합니다. 50%의 확률로 적용됩니다.
            A.RandomResizedCrop(config["input_size"], config["input_size"], p=0.5),
            # 이미지에 탄성 변형을 적용합니다. 50%의 확률로 적용됩니다.
            A.ElasticTransform(p=0.5),
            # 이미지를 압축하여 품질을 조정합니다. 50%의 확률로 적용됩니다.
            A.ImageCompression(quality_lower=85, quality_upper=95, p=0.5),
            # 이미지의 밝기와 대비를 무작위로 조정합니다. 50%의 확률로 적용됩니다.
            A.RandomBrightnessContrast(p=0.5),
            # 블러(Blur) 또는 중앙값 블러(MedianBlur) 중 하나를 무작위로 선택하여 적용합니다. 50%의 확률로 적용됩니다.
            A.OneOf([A.Blur(blur_limit=3), A.MedianBlur(blur_limit=3)], p=0.5),
            # 이미지를 수평으로 뒤집습니다. 50%의 확률로 적용됩니다.
            A.HorizontalFlip(p=1),
            # 이미지의 대비를 제한된 지역에서 증가시킵니다. 50%의 확률로 적용됩니다.
            A.CLAHE(p=1),
            # 이미지의 선명도를 높입니다. 50%의 확률로 적용됩니다.
            A.Sharpen(p=0.5),
            # 이미지를 밝게 만듭니다. 50%의 확률로 적용됩니다.
            A.Brightness(p=0.5),
        ]
    )
