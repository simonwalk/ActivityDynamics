__author__ = 'Simon Walk'
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "simon.walk@tugraz.at"
__status__ = "Development"

import os
import glob

# Little Helper Script to sort extracted files by timestamp
# Requires unix environment!

print 'Starting to sort change-logs'
fnames = glob.glob("../datasets/<dataset>/*.txt")
for fname in fnames:
    cmd = "sort -t ',' -k 1,1 -k 4,4 " + fname + " > " + fname + ".sorted"
    print cmd
    os.system(cmd)