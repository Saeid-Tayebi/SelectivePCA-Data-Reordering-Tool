import numpy as np
from ._lib.pca import PcaClass as pca
import matplotlib.pyplot as plt


def spca(data: np.ndarray, plotting=True, sugg_order: np.ndarray = None):
    """it receives a block of data and sort its columns based on the independent variance seen in each column
    the return block's first column is the most indepedent column and the last column is the most dependent columns 
    if a suggestion order is given to this function, it would sort the block using the suggested list and return how much variance 
    of the entire block of data is covered using each memeber of the suggested order list

    Args:
        data (np.ndarray): original block of data in its already column sorted format
        plotting (bool, optional): plot the variance covered by each column of data. Defaults to True.

    Returns:
        np.ndarray: columns order based on their importance in terms of variance
    """
    sugg_order = sugg_order.copy() if sugg_order is not None else None
    Num_directions = data.shape[1] if sugg_order is None else sugg_order.size
    newColOrder = np.zeros((Num_directions), dtype=int) if sugg_order is None else sugg_order.copy()
    CoveredR2 = np.zeros((Num_directions), dtype=float)
    OriginalIdx = range(Num_directions)
    data_scaled = (data-np.mean(data, axis=0))/np.std(data, axis=0)
    data_scaledOriginal = data_scaled.copy()

    for i in range(Num_directions-1) if sugg_order is None else range(Num_directions):
        if sugg_order is None:
            pcaModel = pca().fit(data_scaled, to_be_scaled=0, n_component=data_scaled.shape[1])
            P = pcaModel.P[:, 0]
            selectedColIdx = np.argmax(np.abs(P))
            # True idx determination
            newColOrder[i] = OriginalIdx[selectedColIdx]
            OriginalIdx = np.delete(OriginalIdx, selectedColIdx)
        else:
            selectedColIdx = sugg_order[i]
            sugg_order[sugg_order > selectedColIdx] -= 1
        YBest = data_scaled[:, selectedColIdx].reshape(-1, 1)
        Ynew = np.delete(data_scaled, selectedColIdx, axis=1)
        PNew = (Ynew.T @ YBest)/(YBest.T @ YBest)
        Enew = Ynew-(YBest @ PNew.T)
        CoveredR2[i] = 1 - (np.sum(np.var(Enew, axis=0)) /
                            np.sum(np.var(data_scaledOriginal, axis=0)))-np.sum(CoveredR2[0:i])

        data_scaled = Enew

    newColOrder[-1] = np.setdiff1d(range(Num_directions), newColOrder[0:-1]
                                   )[0] if sugg_order is None else newColOrder[-1]
    CoveredR2[-1] = (1 - np.sum(CoveredR2)) if sugg_order is None else CoveredR2[-1]

    if plotting is True:
        plt.figure()

        plt.subplot(7, 1, (1, 3))
        xtick_label = ['col'+str(newColOrder[i]+1)
                       for i in range(Num_directions)]
        spca_bar_plot(CoveredR2*100, lim_line=90, xTickLable=xtick_label)

        plt.xlabel('Original Column Number')
        plt.ylabel('Covered Variance (%)')
        plt.title('Variance Covered for Individual Columns(Sorted)')

        plt.subplot(7, 1, (5, 7))
        all_coveredR2 = np.cumsum(CoveredR2)
        xtick_label = [str(i+1)+'col' for i in range(Num_directions)]
        spca_bar_plot(all_coveredR2*100, lim_line=90, xTickLable=xtick_label)

        plt.xlabel('Number of Used Columns')
        plt.ylabel('Covered Variance (%)')
        plt.title('Variance Covered for n Columns')
        plt.show(block=False)
    return newColOrder


def spca_bar_plot(data, sort=None, lim_line=0, newf=False, xTickLable=None):
    data = data.reshape(-1)
    x = range(1, len(data)+1)
    plt.title('Not Sorted Bar PLot')
    if xTickLable is not None:
        plt.xticks(ticks=x, labels=xTickLable)
    if newf is not False:
        plt.figure()
    if sort is True:
        sort_idx = np.argsort(data)
        data = np.sort(data)
        xTickLable = ['com'+str(sort_idx[i]+1) for i in range(len(sort_idx))]
        plt.xticks(ticks=x, labels=xTickLable)
        plt.title('Sorted Bar PLot')
    plt.bar(x, data)
    if lim_line != 0:
        plt.plot([0, len(data)+1], 2*[lim_line],
                 'k--', label='y = ' + str(lim_line))
        plt.legend()
