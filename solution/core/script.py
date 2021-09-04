from argparse import ArgumentParser


def superscript_main(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        '--helpall',
        action='helpall',
        help='List all commands, including advanced ones.',
    )
