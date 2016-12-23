import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_to_data = "../data/"
path_for_output = "../summary/"

skip = True
for root,_,files in os.walk(path_to_data):
    if skip:
        skip=False
        continue
    outputFileName = os.path.join(path_for_output,root,".dat")
    outputData="None"
    print root,
    for f in files:
        #print f
        try:
            data = np.genfromtxt(os.path.join(root,f),skip_header=1,delimiter=",",
                    dtype={"names":("Index","Date","Time","Open","High","Low","Close","Volume","Value"),
                        "formats":("S5","S10","S5","i","i","i","i","i","i")})
        except:
            print "Cought a file error. Skip: %s %s"%(root,f)
            continue
        if type(outputData)==type("None"):
            outputData = data
        else:
            outputData = np.append(outputData,data)
    print len(outputData)

    #print np.unique(outputData["Index"])
    #np.savetxt("test.csv",outputData)
    ofile= root.split("/")[-1]+".hdf5"
    hf = h5py.File(os.path.join(path_for_output, ofile),'w')
    hf.create_dataset("data",data=outputData)
    hf.flush()
    hf.close()
