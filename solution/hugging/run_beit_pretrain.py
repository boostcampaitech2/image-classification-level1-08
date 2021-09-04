import sys
import json
from pathlib import Path


def main():
    content = json.loads(Path(sys.argv[-1]).read_text())
    print(content)


if __name__ == '__main__':
    main()
