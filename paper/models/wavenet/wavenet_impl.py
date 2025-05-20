import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
from models.wavenet import WaveNetModel
from models.wavenet_impl.processors import ENCODERS