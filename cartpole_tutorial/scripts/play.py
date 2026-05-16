"""Script to play the custom cartpole task."""

import sys
from pathlib import Path

# Add the parent directory to sys.path so 'cartpole_tutorial' can be imported
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import the tutorial package. This will automatically register 'Mjlab-Cartpole'
import cartpole_tutorial  # noqa: F401

# Import the main play function from mjlab
from mjlab.scripts.play import main

if __name__ == "__main__":
    main()