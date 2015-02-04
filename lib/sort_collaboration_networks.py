import os
import glob
print('sort merged file')
fnames = glob.glob("../results/graph_sources/collaboration_networks/*.txt")

for fname in fnames:
    cmd = "sort -t ',' -k 1,1 -k 4,4 " + fname + " > " + fname + ".sorted"
    print cmd
    os.system(cmd)