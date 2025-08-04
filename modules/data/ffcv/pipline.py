import torch

import torchvision.transforms as transforms
import ffcv.transforms as fftransforms
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder
from .operation import DivideImage255

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

tinyimagenet_mean = (0.4802, 0.4481, 0.3975)
tinyimagenet_std = (0.2770, 0.2691, 0.2821)

def pipline_dispatch(query: str, device):
    if query == 'cifar100_train_1':
        pipline = [
            SimpleRGBImageDecoder(),
            fftransforms.RandomTranslate(padding=4),
            fftransforms.RandomHorizontalFlip(flip_prob=.5),
            fftransforms.RandomColorJitter(brightness=63 / 255),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
        ]
    elif query == 'cifar100_train_2':
        pipline = [
            SimpleRGBImageDecoder(),
            fftransforms.RandomTranslate(padding=4),
            fftransforms.RandomHorizontalFlip(flip_prob=.5),
            fftransforms.RandomColorJitter(brightness=63 / 255),
            fftransforms.Cutout(16),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        ]
    elif query == 'cifar100_test_1':
        pipline = [
            SimpleRGBImageDecoder(),
            fftransforms.RandomHorizontalFlip(flip_prob=0.), 
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
        ]
    elif query == 'imagenet_train_1':
        pipline = [
            RandomResizedCropRGBImageDecoder((224, 224)),
            fftransforms.RandomHorizontalFlip(),
            fftransforms.RandomColorJitter(brightness=63 / 255),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
        ]
    elif query == 'imagenet_train_2':
        pipline = [
            RandomResizedCropRGBImageDecoder((224, 224)),
            fftransforms.RandomHorizontalFlip(),
            fftransforms.RandomColorJitter(brightness=63 / 255),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        ]
    elif query == 'imagenet_test_1':
        pipline = [
            RandomResizedCropRGBImageDecoder((224, 224), (1,1), (1,1)),
            fftransforms.RandomHorizontalFlip(flip_prob=0.),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage()
        ]
    elif query == 'tinyimagenet_train_1':
        pipline = [
            RandomResizedCropRGBImageDecoder((64, 64)),
            fftransforms.RandomHorizontalFlip(),
            fftransforms.RandomColorJitter(brightness=63 / 255),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
        ]
    elif query == 'tinyimagenet_train_2':
        pipline = [
            RandomResizedCropRGBImageDecoder((64, 64)),
            fftransforms.RandomHorizontalFlip(),
            fftransforms.RandomColorJitter(brightness=63 / 255),
            ImageNetPolicy(),  # Assuming this is defined elsewhere
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
        ]
    elif query == 'tinyimagenet_test_1':
        pipline = [
            ResizeRGBImageDecoder((64, 64)),
            CenterCropRGBImageDecoder((64, 64)),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
        ]

    pipline.extend([
        fftransforms.ToDevice(torch.device(device)),
        fftransforms.Convert(torch.float32),
        DivideImage255(),
    ])

    if query.startswith('cifar100'):
        pipline.append(transforms.Normalize(mean=cifar100_mean, std=cifar100_std, inplace=True))
    elif query.startswith('imagenet'):
        pipline.append(transforms.Normalize(mean=imagenet_mean, std=imagenet_std, inplace=True))
    elif query.startswith('tinyimagenet'):
        pipline.append(transforms.Normalize(mean=tinyimagenet_mean, std=tinyimagenet_std, inplace=True))

    label_pipeline = [
        IntDecoder(), 
        fftransforms.ToTensor(),
        fftransforms.ToDevice(torch.device(device))
    ]

    return pipline, label_pipeline