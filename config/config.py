from __future__ import division

__author__ = 'Simon Walk, Florian Geigl, Denis Helic'
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "simon.walk@tugraz.at"
__status__ = "Development"

import os
from collections import defaultdict
import datetime

from numpy.random.mtrand import poisson
from graph_tool.all import *
from numpy.linalg import eig
import numpy as np
from multiprocessing import Pool

# Change if necessary
r_binary_path = '/usr/bin/RScript'

# Default folders are set to create plots within results folder
base_dir = "results/"
graph_binary_dir = base_dir + "graph_binaries/"
graph_source_dir = base_dir + "graph_sources/"
ds_source_dir = "datasets/"
plot_dir = base_dir + "plots/"
graph_dir = base_dir + "graphs/"
r_dir = "R Scripts/"
