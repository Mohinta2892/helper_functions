import numpy as np
import matplotlib.pyplot as plt
# Load skeletons
skeletons = np.load("neuron_skeletons.npy", allow_pickle=True).item()
all_skids = list(skeletons.keys())
# Get a specific skeleton
cur_skid = all_skids[100]
cur_skeleton = np.array(skeletons[cur_skid])[0,:,:]
# Plot skeletonplt.close("all")
fig,ax = plt.subplots(1,3,figsize=(15,5))
ax[0].scatter(cur_skeleton[:,0], cur_skeleton[:,1], c="k")
ax[1].scatter(cur_skeleton[:,0], cur_skeleton[:,2], c="k")
ax[2].scatter(cur_skeleton[:,1], cur_skeleton[:,2], c="k")
plt.savefig("tmp") 
