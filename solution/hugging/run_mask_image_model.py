import os
import sys

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig

from dataclasses import dataclass, field, asdict

from .huggingtemplate import HfArgumentParser
from .huggingtemplate.args import (
    BeitPretrainDataArguments,
    BeitPretrainTrainingArguments,
    BeitPretrainModelArguments,
    BeitPretrainCollateArguments,
    # BeitPretrainMetricArguments,
    # BeitPretrainAlarmArguments,
)
from .huggingtemplate import collate_funcs
from .huggingtemplate import datasets
# from huggingtemplate import metrics
from .huggingtemplate import models
from .huggingtemplate import trainers

from dall_e_tok import DALLETokenizer


def main():
    parser = HfArgumentParser(
        (
            BeitPretrainDataArguments,
            BeitPretrainTrainingArguments,
            BeitPretrainModelArguments,
            BeitPretrainCollateArguments,
            # BeitPretrainMetricArguments,
            # BeitPretrainAlarmArguments,
         )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # read args from json file
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        # read args from shell script or real arguments
        args = parser.parse_args_into_dataclasses()

    data_args, training_args, model_args, collate_args = args

    if training_args.report_to == "wandb":
        import wandb
        wandb.login()
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    dataset_cls = getattr(datasets, data_args.dataset_class)
    trainer_cls = getattr(trainers, training_args.trainer_class)
    collate_fn = getattr(collate_funcs, collate_args.collate_fn)
    # metric_fn = getattr(metrics, metric_args.metric_fn)

    transform = collate_fn(collate_args)

    train_dataset = dataset_cls.load(
        data_args.train_data_dir,
        transform=transform,
    )

    for aug_data_path in data_args.augmented_data_dir:
        aug_dataset = dataset_cls.load(
            aug_data_path,
            transform=transform,
        )
        train_dataset += aug_dataset

    tokenizer = DALLETokenizer.from_pretrained("jinmang2/dall-e-tokenizer")

    def model_init():
        import transformers
        transformers.logging.set_verbosity_warning()
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model_class = getattr(models, model_args.architectures)
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        return model

    trainer = trainer_cls(
        dall_e_tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        model_init=model_init,
    )

    trainer.train()
    if training_args.report_to == "wandb":
        wandb.finish()

    trainer.model.save_pretrained(
        save_directory=os.path.join(training_args.output_dir, training_args.run_name),
    )

if __name__ == "__main__":
    main()
