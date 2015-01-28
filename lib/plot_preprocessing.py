from collections import defaultdict
import datetime
import numpy as np
import os as os
import shutil
import pandas as pd
from pandas.util.testing import Series

percentage_experiments = ["_PERC_RAND", "_PERC_SPEC"]
percentage = np.arange(0.0, 1.01, 0.01)

is_notebook = False
is_externalHDD = not is_notebook

if is_notebook:
    g_path = "/Volumes/StorageDisk/"
if is_externalHDD:
    g_path = "/Volumes/MyBook4TB/"

source = g_path + "DynamicNetworks/"

def remove_file(fname):
    try:
        os.remove(fname)
    except:
        pass

#Karate     => 0.015
#PA         => 0.008
#R          => 0.008
#SBM ASC    => 0.013
#SBM DC     => 0.013

deltatau_dict = {"Karate_0PERC_RAND":0.015, "Karate_0PERC_SPEC": 0.015, "PrefAttach_MEDIUM_0PERC_RAND": 0.008,
                 "PrefAttach_MEDIUM_0PERC_SPEC": 0.008, "Random_MEDIUM_0PERC_RAND":0.008,
                 "Random_MEDIUM_0PERC_SPEC":0.008, "SBM_ASC_SAME_CONN_0PERC_RAND":0.013,
                 "SBM_ASC_SAME_CONN_0PERC_SPEC":0.013, "SBM_DEG_CORR_0PERC_RAND":0.013, "SBM_DEG_CORR_0PERC_SPEC":0.013}
graph_names = ["Karate_0PERC_RAND", "Karate_0PERC_SPEC", "PrefAttach_MEDIUM_0PERC_RAND",
               "PrefAttach_MEDIUM_0PERC_SPEC", "Random_MEDIUM_0PERC_RAND", "Random_MEDIUM_0PERC_SPEC",
               "SBM_DEG_CORR_0PERC_RAND", "SBM_DEG_CORR_0PERC_SPEC"]

ratios = [x for x in xrange(0, 121, 10) if x > 0]
ratios.reverse()
ratios.append(1)

new_t = pd.DataFrame()
new_dt = pd.DataFrame()
for ratio in ratios:
    ratiostring = str(float(ratio)).replace(".", "")
    for graph_name in graph_names:
        deltatau = deltatau_dict[graph_name]
        deltatau_string = str(deltatau).replace(".", "")
        t = pd.read_csv(source + "graph_sources/weights/"+graph_name +"/" + graph_name + "_10000_iterations_"+ratiostring+"_"+deltatau_string+"_avg.txt", sep="\t", header=None)
        t = t.mean()
        new_t[graph_name] = t
        new_dt[graph_name] = Series([x*deltatau for x in t.index], index=t.index)

    print new_t
    new_t.to_csv('average_activity_over_tau_'+str(ratiostring)+'_activity.csv')
    new_t.to_csv('average_activity_over_tau_'+str(ratiostring)+'_tau.csv')
