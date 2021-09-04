import os
import sys
import json
from pathlib import Path


def main():
    file = json.loads(Path(sys.argv[-1]).read_text(encoding="utf-8"))
    print(file)


if __name__ == '__main__':
    main()
