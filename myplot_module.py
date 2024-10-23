
import numpy as np
import matplotlib.pyplot as plt


def BarPlot(data,sort=None,lim_line=0,newf=False,xTickLable=None):
    '''
    data can be in 1D or 0D,
    it sorted is ask then it would sort both x tick and the data altogether
    '''
    data=data.reshape(-1)
    x=range(1,len(data)+1)
    plt.title('Not Sorted Bar PLot')
    if xTickLable is not None:
        plt.xticks(ticks=x,labels=xTickLable)
    if newf is not False:
        plt.figure()
    if sort is True:
        sort_idx=np.argsort(data)
        data=np.sort(data)
        xTickLable=['com'+str(sort_idx[i]+1) for i in range(len(sort_idx))]
        plt.xticks(ticks=x,labels=xTickLable)
        plt.title('Sorted Bar PLot')
    plt.bar(x,data)  
    if lim_line !=0:
        plt.plot([0,len(data)+1],2*[lim_line],'k--',label='y = '+ str(lim_line))
        plt.legend()

