
import numpy as np
from spca.spca import spca

OriginalData = np.array([[0.32449153, -0.37284353,  0.20279008, -0.27719296,  0.14349545],
                         [0.33884964,  0.22548439, -0.00117191, -0.81653025, -0.00562129],
                         [-0.03747781,  0.41583203,  0.09654524, -0.55816716, -0.27632115],
                         [0.35052732,  0.16028855, -0.26095015, -0.47245232,  0.20200274],
                         [0.4649153,  0.06998226,  0.00783886, -0.83527089,  0.10163089],
                         [0.643428, -0.47255018,  0.01768985, -0.46102387,  0.4111165],
                         [0.55044962, -0.55239036, -0.25167061,  0.07820573,  0.60052114],
                         [0.42278671, -0.42982882, -0.06171441, -0.07605556,  0.37649641],
                         [0.15586103, -0.01676184,  0.15191581, -0.39260655, -0.04569263],
                        [0.30366606,  0.18603854,  0.05744663, -0.77552741, -0.03616079]])

# Sorting based on the data itself
newColOrder = spca(data=OriginalData, plotting=True)


# Sorting based on the suggested itself
newColOrder = spca(data=OriginalData, plotting=True, sugg_order=np.array([3, 4]))
