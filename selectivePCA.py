import numpy as np
from MyPcaClass import MyPca as pca
import matplotlib.pyplot as plt
import myplot_module

def SPCA(data:np.ndarray,plotting=True):
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

    newColOrder[-1] = np.setdiff1d(range(Num_directions),newColOrder[0:-1])
    CoveredR2[-1] = 1-np.sum(CoveredR2)
    organized_data=dataOrigin[:,newColOrder]

    if plotting is True:
        plt.figure()
        
        plt.subplot(7,1,(1,3))
        xtick_label=['col'+str(newColOrder[i]+1) for i in range(Num_directions)]
        myplot_module.BarPlot(CoveredR2*100,lim_line=90,xTickLable=xtick_label)

        plt.xlabel('Original Column Number')
        plt.ylabel('Covered Variance (%)')
        plt.title('Variance Covered for Individual Columns(Sorted)')

        plt.subplot(7,1,(5,7))
        all_coveredR2 = np.cumsum(CoveredR2)
        xtick_label=[str(i+1)+'col' for i in range(Num_directions)]
        myplot_module.BarPlot(all_coveredR2*100,lim_line=90,xTickLable=xtick_label)
  
        plt.xlabel('Number of Used Columns')
        plt.ylabel('Covered Variance (%)')
        plt.title('Variance Covered for n Columns')
        plt.show(block=False)
    return newColOrder,organized_data

