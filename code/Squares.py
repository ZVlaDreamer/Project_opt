import matplotlib.pyplot as plt
plt.rc("text", usetex=True)
import ipywidgets as ipywidg
import numpy as np
import liboptpy.unconstr_solvers as methods
import liboptpy.step_size as ss
import math
import time
import seaborn as sns
import scipy.optimize as optimize
%matplotlib inline
import matplotlib.pylab as pylab

from Current_oracle import *
from GKLBi_diagonalization import GKL_
