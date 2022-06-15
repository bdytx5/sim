import os, sys

# import parent directory 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
from pathlib import Path

import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import strip_optimizer

## FOR IF MODEL TRAINING WAS ENDED EARLY 
strip_optimizer('/Users/brett/Desktop/sim/game/models/best.pt')
