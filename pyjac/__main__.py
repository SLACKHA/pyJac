import sys

from .pyjac import pyJac


def main(args=None):
    if args is None:
        args = sys.argv[1:]
        pyJac(args)


if __name__ == '__main__':
    sys.exit(main())
