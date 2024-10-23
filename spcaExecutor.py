import numpy as np
import selectivePCA

# Generating data
Num_observation=30
Ninput=5
Noutput=10

X =np.random.rand(Num_observation,Ninput)
Beta=np.random.rand(Ninput,Noutput) * 2 -1 
Y=(X @ Beta)
new_col,organized_data=selectivePCA.SPCA(Y)