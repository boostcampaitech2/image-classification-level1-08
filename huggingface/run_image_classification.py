import os
import sys

import torch
import torch.nn as nn
import transformers

import pandas as pd
from dataclasses import asdict

from transformers import AutoConfig, AutoFeatureExtractor

import models
from data import FaceMaskDataset
from transform import build_transform
from args import (
    DataArguments,
    TrainingArguments,
    ModelArguments,
    TransformArguments,
    MyArgumentParser,
)
from metrics import make_compute_metrics
from trainer import Trainer


def main():
    parser = MyArgumentParser((DataArguments, TrainingArguments, ModelArguments, TransformArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # read args from json file
        data_args, training_args, model_args, transform_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        # read args from shell script or real arguments
        data_args, training_args, model_args, transform_args = parser.parse_args_into_dataclasses()

    if training_args.report_to == "wanbd":
        import wandb
        wandb.login()
        %env WANDB_PROJECT={training_args.wandb_project}

    transform = build_transform(transform_args)

    # @TODO: argument parsing
    dataset = FaceMaskDataset.load(
        data_args.train_data_dir,
        is_train=True,
        transform=transform,
        return_image=data_args.return_image,
        level=data_args.level,
    )

    # @TODO: 팀원들과 공유된 validation set을 사용할 것
    train_dataset, eval_dataset = dataset.train_test_split(0.2)

    compute_metrics = make_compute_metrics(data_args.level)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    def model_init():
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        for key, value in asdict(model_args).items():
            setattr(config, key, value)
        model_class = getattr(models, ModelArguments.architecture)
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
        )
        return model

    trainer = Trainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init,
    )

    trainer.train()

    # @TODO: 자동화
    trainer.model.save_pretrained(
        save_directory=os.path.join(training_args.output_dir, training_args.run_name),
    )

    test_dataset = FaceMaskDataset.load(
        data_args.test_data_dir,
        is_train=False,
        transform=feature_extractor,
        return_image=data_args.return_image,
        level=data_args.level,
    )

    # @TODO: level 3 script
    # @TODO: transform 처리 자동화
    predictions = trainer.predict(test_dataset=test_dataset)
    preds = predictions.predictions
    preds = preds.argmax(axis=-1)

    # @TODO: submission 생성 자동화
    test_dir = "../input/data/eval/"
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))

    file2label = dict(zip(data.total_imgs, preds.tolist()))
    submission['ans'] = submission.ImageID.map(file2label)

    submission.to_csv(
        os.path.join(test_dir, "submission-" + training_args.run_name + ".csv"),
        index=False
    )


if __name__ == "__main__":
    main()
