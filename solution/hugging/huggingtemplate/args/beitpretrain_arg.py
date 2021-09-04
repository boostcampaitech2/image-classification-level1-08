from typing import List
from dataclasses import dataclass, field
from .base import (
    DataArguments,
    TrainingArguments,
    ModelArguments,
    CollateArguments,
    MetricArguments,
    AlarmArguments,
)
from .argparse import lambda_field


@dataclass
class BeitPretrainDataArguments(DataArguments):
    dataset_class: str = field(default="BeitPretrainDataset")
    train_data_dir: str = field(default="new_standard.csv")
    augmented_data_dir: List[str] = lambda_field(default=[])


@dataclass
class BeitPretrainTrainingArguments(TrainingArguments):
    trainer_class: str = field(default="BeitForMIMTrainer")
    wandb_project: str = field(default='beit_pretrain')
    report_to: str = field(default='wandb')
    run_name: str = field(default='test')


@dataclass
class BeitPretrainModelArguments(ModelArguments):
    model_name_or_path: str = field(default="microsoft/beit-large-patch16-224-pt22k")
    cache_dir: str = field(default='cache')
    architectures: str = field(default='BeitForMaskedImageModeling')
    transformers_version: str = field(default='4.10.0.dev0')


@dataclass
class BeitPretrainCollateArguments(CollateArguments):
    collate_fn: str = field(default='DataAugmentationForBEiT')
    input_size: int = field(default=224)
    second_input_size: int = field(default=112)
    color_jitter: float = field(default=0.4)
    imagenet_default_mean_and_std: bool = field(default=False)
    discrete_vae_type: str = field(default='dall-e')
    patch_size: int = field(default=16)
    num_mask_patches: int = field(default=75)
    max_mask_patches_per_block: int = field(default=None)
    min_mask_patches_per_block: int = field(default=16)
    train_interpolation: str = field(default='bicubic')
    second_interpolation: str = field(default='lanczos')

    def __post_init__(self):
        self.window_size = (self.input_size // self.patch_size,) * 2


@dataclass
class BeitPretrainMetricArguments(MetricArguments):
    metric_fn: str = field(default=None)


@dataclass
class BeitPretrainAlarmArguments(AlarmArguments):
    alarm_type: str = field(default=None)
