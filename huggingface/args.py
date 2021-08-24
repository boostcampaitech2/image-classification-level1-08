import os
import re
import sys
import copy
import json
from pathlib import Path
from enum import Enum
import dataclasses
from dataclasses import dataclass, field
from argparse import ArgumentParser, ArgumentTypeError
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union

from transformers.training_args import TrainingArguments


DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def lambda_field(default, **kwargs):
    return field(default_factory=lambda: copy.copy(default))


@dataclass
class DataArguments:
    train_data_dir: str = field(default="../input/data/train/images/")
    test_data_dir: str = field(default="../input/data/eval/images/")
    return_image: bool = field(default=False)
    level: int = field(default=1)


@dataclass
class MyTrainingArguments(TrainingArguments):
    wandb_project: str = field(default='deit-mask-face')
    report_to: str = field(default='wandb')
    run_name: str = field(default='test')
    load_best_model_at_end: bool = field(default=True)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default='facebook/deit-base-distilled-patch16-224')
    cache_dir: str = field(default='cache')
    architecture: str = field(default='DeiTForSingleHeadClassification')
    num_labels: int = field(default=18)
    num_mask_labels: int = field(default=3)
    num_gender_labels: int = field(default=2)
    num_age_labels: int = field(default=3)
    transformers_version: str = field(default='4.10.0.dev0')
    epsilon: float = field(default=0.1)

    def __post_init__(self):
        self.label2id = {str(i): i for i in range(self.num_labels)}
        self.id2label = {i: str(i) for i in range(self.num_labels)}


@dataclass
class TransformArguments:
    is_train: bool = field(default=True)
    input_size: int = field(default=224)
    color_jitter: float = field(default=0.4)
    aa: str = field(default='rand-m9-mstd0.5-inc1')
    train_interpolation: str = field(default='bicubic')
    reprob: float = field(default=0.25)
    remode: str = field(default='pixel')
    recount: int = field(default=1)
    rand_ratio: float = field(default=0.45)


class MyArgumentParser(ArgumentParser):
    """
    See @ https://github.com/huggingface/transformers/blob/master/src/transformers/hf_argparser.py
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released."
                )
            typestring = str(field.type)
            for prim_type in (int, float, str):
                for collection in (List,):
                    if (
                        typestring == f"typing.Union[{collection[prim_type]}, NoneType]"
                        or typestring == f"typing.Optional[{collection[prim_type]}]"
                    ):
                        field.type = collection[prim_type]
                if (
                    typestring == f"typing.Union[{prim_type.__name__}, NoneType]"
                    or typestring == f"typing.Optional[{prim_type.__name__}]"
                ):
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = [x.value for x in field.type]
                kwargs["type"] = type(kwargs["choices"][0])
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                else:
                    kwargs["required"] = True
            elif field.type is bool or field.type == Optional[bool]:
                if field.default is True:
                    self.add_argument(
                        f"--no_{field.name}", action="store_false", dest=field.name, **kwargs)

                # Hack because type=bool in argparse does not behave as we want.
                kwargs["type"] = string_to_bool
                if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                    # Default value is True if we have no default when of type bool.
                    default = True if field.default is dataclasses.MISSING else field.default
                    # This is the value that will get picked if we don't include --field_name in any way
                    kwargs["default"] = default
                    # This tells argparse we accept 0 or 1 value after --field_name
                    kwargs["nargs"] = "?"
                    # This is the value that will get picked if we do --field_name (without value)
                    kwargs["const"] = True
            elif (
                hasattr(field.type, "__origin__") and re.search(
                    r"^typing\.List\[(.*)\]$", str(field.type)) is not None
            ):
                kwargs["nargs"] = "+"
                kwargs["type"] = field.type.__args__[0]
                assert all(
                    x == kwargs["type"] for x in field.type.__args__
                ), "{} cannot be a List of mixed types".format(field.name)
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                elif field.default is dataclasses.MISSING:
                    kwargs["required"] = True
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(
        self, args=None, return_remaining_strings=False, look_for_args_file=True, args_filename=None
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass types.
        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args
        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses this file instead of the ".args" file specified in the previous argument.
        Returns:
            Tuple consisting of:
                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + \
                    args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(
                    f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}")

            return (*outputs,)

    def parse_json_file(self, json_file: str) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)

    def parse_dict(self, args: dict) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass
        types.
        """
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in args.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)
