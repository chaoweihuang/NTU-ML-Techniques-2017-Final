import sys

from utility import *

pred = np.array([float(line.strip()) for line in open(sys.argv[1])])
write_submit_file(sys.argv[1]+'.binary', (pred>float(sys.argv[2])).astype(int))
