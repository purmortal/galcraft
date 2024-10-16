import numpy as np
import pytest
from pathlib import Path

# Define the directories for the test files
test_dir = Path(__file__).parent
true_dir = test_dir / "outputs"
outputs_dir = test_dir / "truth"

print(outputs_dir, true_dir)
