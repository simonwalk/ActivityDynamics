from __future__ import division

__author__ = 'Simon Walk'

import os
from collections import defaultdict
import datetime

from numpy.random.mtrand import poisson
from graph_tool.all import *
from numpy.linalg import eig
import numpy as np
from multiprocessing import Pool

base_dir = "../DynamicNetworksResults/"

graph_binary_dir = base_dir + "graph_binaries/"
graph_source_dir = base_dir + "graph_sources/"
plot_dir = base_dir + "plots/"
graph_dir = base_dir + "graphs/"
r_dir = "../DynamicNetworks/R Scripts/"