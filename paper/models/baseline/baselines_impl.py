import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
from models.benchmark import RollingAverageWeek, AutoRollingAverageWeek
from models.exp_rolling_average_week import ExpWeightedRollingWeek