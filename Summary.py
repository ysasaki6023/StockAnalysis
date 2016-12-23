import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_for_input  = "../summary/"
outputFileName  = "all.hdf5"

OutputData = "None"

skip = True
for root,_,files in os.walk(path_for_input):
    for f in files:
        print f
        hf = h5py.File(os.path.join(path_for_input,f),"r")
        data = hf["data"].value
        print np.unique(data["Index"])
        #print data
        if type(OutputData) == type("None"):
            OutputData = data
        else:
            OutputData = np.append(OutputData,data)

print len(OutputData)
hf = h5py.File(outputFileName,'w')
hf.create_dataset("data",data=OutputData)
hf.flush()
hf.close()
