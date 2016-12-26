import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# Settings
NumberOfStocks = 5

ff = h5py.File("all.hdf5","r")
print "loaded"
#print ff["data"].value

d = pd.DataFrame(ff["data"].value)
print "DF ready"
d["wDatetime"] = d["Date"]+" "+d["Time"]
d["tDatetime"] = pd.to_datetime(d["wDatetime"])
print "Datetime converted"
#d["tDate"]     = pd.to_datetime(d["Date"])
#d["tTime"]     = pd.to_datetime(d["Time"])
print d
#d.set_index("tDatetime")
d = d.set_index(pd.DatetimeIndex(d["tDatetime"]))
d = d.sort_values(by=["Index","tDatetime"])
print "Sorted"
for k in ["Open","High","Low","Close"]:
    print k
    d.loc[d[k] == -1] = np.nan
d=d.fillna(method="ffill")

tops=d.groupby("Index").mean().sort_values("Volume",ascending=False)
tops=tops[:NumberOfStocks]
tops_index = tops["Index"].unique()
dd = d[d["Index"]==tops_index]

#dd = d.pivot(index="tDatetime",columns="Index",values="Close")
##dd = pd.pivot_table(d,index=["Date","Time"],columns="Index",values="Close")
dd = pd.pivot_table(dd,index="tDatetime",columns="Index",values="Close")
#dd = dd.sort_values(by="tDatetime")
dd=dd.fillna(method="ffill")
dd=dd.fillna(method="bfill")
print dd
dd.to_csv("test.csv")
print "done"
