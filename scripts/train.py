"""Mythos Go2 학습 entry (cartpole_tutorial/scripts/train.py 패턴).

repo 루트를 sys.path 에 넣고 task 패키지를 import → register_mjlab_task 실행.
scripts/ 는 패키지 아님 (mjlab/원 Mythos 스켈레톤 관례).

  conda run --no-capture-output -n mjlab_env python scripts/train.py Mythos-Go2-WBC
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo 루트

import locomotion.config.go2 as _go2  # noqa: E402

from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  _go2.register()  # cfg stub 구현 완료 후 정상 동작 (지금은 NotImplementedError)
  main()
