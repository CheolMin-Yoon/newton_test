# Root init to make the directory importable
import sys
from pathlib import Path

# Add this directory to the python path so the inner modules can import each other easily
current_dir = str(Path(__file__).parent.resolve())
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Register the task when this package is imported
from .tasks import cartpole
