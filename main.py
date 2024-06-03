import os
import yaml
import pickle
import meep as mp
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import get_slice
import time
import h5py

import simulation
import field_monitors
import argparse

from utils.general import load_config, parse_args

def run(params):

    

if __name__=="__main__":

    params = load_config(sys.args)    

    run(params)
