#!/usr/bin/env python3
"""Re-record the golden generated-source fixtures.

Run this ONLY when generated output is intentionally changing. Review the
resulting diff carefully -- it is the record of exactly how the generated
Jacobian and rate source changed -- and note the change in CHANGELOG.md.

    python test/regenerate_golden.py
"""

import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from conftest import GOLDEN_DIR, GOLDEN_MECHS, GOLDEN_VARIANTS  # noqa: E402

from pyjac.core.create_jacobian import create_jacobian  # noqa: E402


def main():
    for name, mech in sorted(GOLDEN_MECHS.items()):
        if not mech.is_file():
            sys.exit(f'missing mechanism: {mech}')
        for variant, (lang, kwargs) in sorted(GOLDEN_VARIANTS.items()):
            dest = GOLDEN_DIR / name / variant
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            create_jacobian(lang, mech_name=str(mech), build_path=str(dest), **kwargs)
            count = sum(1 for p in dest.rglob('*') if p.is_file())
            print(f'recorded {name}/{variant}: {count} files')


if __name__ == '__main__':
    main()
