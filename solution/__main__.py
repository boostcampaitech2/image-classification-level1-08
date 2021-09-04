import sys
from importlib import import_module
from argparse import ArgumentParser


HUGGING_RUN_SCRIPTS = {
    "image_classification": "run_image_classification",
    "mask_image_modeling": "run_mask_image_model",
    "test_module": "run_test_module"
}

JISOO_RUN_SCRIPTS = {
    "train": "train",
    "test": "test",
}

CNN_ENGINE_RUN_SCRIPTS = {
    "train": "train",
}

MOON_ENGINE_RUN_SCRIPTS = {
    "train": "train",
}

LIBRARY_MAP = {
    "hugging": HUGGING_RUN_SCRIPTS,
    "jisoo": JISOO_RUN_SCRIPTS,
    "cnn_engine": CNN_ENGINE_RUN_SCRIPTS,
    "moon": MOON_ENGINE_RUN_SCRIPTS,
}


def main(args: ArgumentParser):
    script_list = LIBRARY_MAP.get(args.module, None)
    if script_list is None:
        raise AttributeError

    module = import_module(args.module)
    script_name = script_list.get(args.script, None)

    script = getattr(module, script_name)
    sys.argv = sys.argv[-2:]
    if hasattr(script, 'main'):
        script.main()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--module', default='hugging')
    parser.add_argument('-s', '--script', default='image_classification')
    parser.add_argument('-c', '--config')
    args = parser.parse_args()
    main(args)
