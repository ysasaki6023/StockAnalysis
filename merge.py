import h5py
import os, glob

mylist = ["lin.W.W","lin.W.b","GRU.W.W","GRU.W.b","GRU.U.W","GRU.U.b","GRU.U_r.W","GRU.U_r.b","GRU.U_z.W","GRU.U_z.b","GRU.W_r.W","GRU.W_r.b","GRU.W_z.W","GRU.W_z.b"]

outfile = h5py.File("params.hdf5","w")
for f in glob.glob("result/*.hdf5"):
    print(f)
    with h5py.File(f,"r") as infile:
        ff = f.replace("result/snapshot_iter_","").replace(".hdf5","")
        outfile.create_group(ff)
        for n in mylist:
            outfile.create_dataset(ff+"/"+n,data=infile[n])
