import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
# import torchvision.transforms as transforms

norm_mean=(0.4914, 0.4822, 0.4465) 
norm_std=(0.2023, 0.1994, 0.2010)

train_transforms = A.Compose(
    [
    A.Sequential([
        A.PadIfNeeded(
            min_height=40,
            min_width=40,
            border_mode=cv2.BORDER_CONSTANT,
            value=(norm_mean[0]*255, norm_mean[1]*255, norm_mean[2]*255)
        ),
        A.RandomCrop(
            height=32,
            width=32
        )
    ], p=1),
    A.CoarseDropout(
            max_holes=2,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=tuple((x * 255.0 for x in norm_mean)),
            p=0.8,
        ),
    A.Normalize(norm_mean, norm_std),
    ToTensorV2()
]
)

test_transforms = A.Compose(
    [
    A.Normalize(norm_mean, norm_std, always_apply=True),
    ToTensorV2()
]
)