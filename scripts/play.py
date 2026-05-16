"""Mythos Go2 평가 entry (cartpole_tutorial/scripts/play.py 패턴).

  conda run --no-capture-output -n mjlab_env python scripts/play.py Mythos-Go2-WBC --viewer native
  (viser 면 --no-capture-output 필수 — 링크 로그가 conda run 에 캡처됨)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo 루트

import locomotion.config.go2 as _go2  # noqa: E402

from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
  _go2.register()
  main()
