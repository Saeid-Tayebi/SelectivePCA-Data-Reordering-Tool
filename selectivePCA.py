#%%
import numpy as np
import matplotlib.pyplot as plt
from MyPcaClass import MyPca as pca

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

def SPCA(data:np.ndarray):
    '''
    it receives a block of data and sort its columns based on the independent variance seen in each column
    the return block's first column is the most indepedent column and the last column is the most dependent columns 
    '''
    Num_directions=data.shape[1]
    newColOrder=np.zeros((Num_directions),dtype=int)
    CoveredR2=np.zeros((Num_directions))
    OriginalIdx=range(Num_directions)
    dataOrigin=data.copy()
    data_scaled=(data-np.mean(data,axis=0))/np.std(data,axis=0)
    data_scaledOriginal=data_scaled.copy()

    for i in range(Num_directions-1):
        pcaModel = pca()
        pcaModel.train(data_scaled,to_be_scaled=0,Num_com=data_scaled.shape[1])
        P=pcaModel.P[:,0]
        selectedColIdx  =np.argmax(np.abs(P))
        YBest = data_scaled[:,selectedColIdx].reshape(-1,1)
        Ynew = np.delete(data_scaled, selectedColIdx, axis=1)
        PNew=(Ynew.T @ YBest)/(YBest.T @ YBest)
        Enew=Ynew-(YBest @ PNew.T)
        CoveredR2[i] = 1 - (np.sum(np.var(Enew,axis=0))/np.sum(np.var(data_scaledOriginal,axis=0)))-np.sum(CoveredR2[0:i])

        data_scaled=Enew
        # True idx determination
        newColOrder[i]=OriginalIdx[selectedColIdx]
        OriginalIdx=np.delete(OriginalIdx,selectedColIdx)

    newColOrder[-1] = np.setdiff1d(range(Num_directions),newColOrder[0:-1])[0]
    CoveredR2[-1] = 1-np.sum(CoveredR2)
    organized_data=dataOrigin[:,newColOrder]

    plt.figure()
    
    plt.subplot(7,1,(1,3))
    xtick_label=['col'+str(newColOrder[i]+1) for i in range(Num_directions)]
    BarPlot(CoveredR2*100,lim_line=90,xTickLable=xtick_label)

    plt.xlabel('Sorted Original Column Numbers')
    plt.ylabel('Covered Variance (%)')
    plt.title('Variance Covered for Individual Columns(Sorted)')

    plt.subplot(7,1,(5,7))
    all_coveredR2 = np.cumsum(CoveredR2)
    xtick_label=[str(i+1)+'col' for i in range(Num_directions)]
    BarPlot(all_coveredR2*100,lim_line=90,xTickLable=xtick_label)

    plt.xlabel('Cumulative Columns Selected')
    plt.ylabel('Covered Variance (%)')
    plt.title('Variance Covered for n Columns')
    plt.show(block=False)
    return newColOrder , CoveredR2 , organized_data  

# Code execution
OriginalData=np.array([ [ 0.32449153, -0.37284353,  0.20279008, -0.27719296,  0.14349545],
                        [ 0.33884964,  0.22548439, -0.00117191, -0.81653025, -0.00562129],
                        [-0.03747781,  0.41583203,  0.09654524, -0.55816716, -0.27632115],
                        [ 0.35052732,  0.16028855, -0.26095015, -0.47245232,  0.20200274],
                        [ 0.4649153 ,  0.06998226,  0.00783886, -0.83527089,  0.10163089],
                        [ 0.643428  , -0.47255018,  0.01768985, -0.46102387,  0.4111165 ],
                        [ 0.55044962, -0.55239036, -0.25167061,  0.07820573,  0.60052114],
                        [ 0.42278671, -0.42982882, -0.06171441, -0.07605556,  0.37649641],
                        [ 0.15586103, -0.01676184,  0.15191581, -0.39260655, -0.04569263],
                        [ 0.30366606,  0.18603854,  0.05744663, -0.77552741, -0.03616079]])

new_col_order , CoveredR2 , OrganizedData =SPCA(OriginalData)

# %%
