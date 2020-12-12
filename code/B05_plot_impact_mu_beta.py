import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def plot_contour(data, save_fig):
    col = np.linspace(0.1, 100, num=1000)
    # col = np.log10(col)
    col = np.round(col,1)
    col2 = np.linspace(10, 0.01, num=1000)
    # # col = np.log10(col)
    col2 = np.round(col2,2)

    # data.columns = col
    # data.index =col2
    # data.columns = col
    # data['beta_I'] = col

    # data = pd.melt(data,id_vars=['beta_I'],value_vars = col)
    # data = data.rename(columns = {'variable':'mu','value':'RO'})

    font_size = 16
    matplotlib.rcParams['font.size'] = font_size - 2
    fig, ax = plt.subplots(1, 1, figsize=[9, 7])

    contours = plt.contour(col, col, data.values, [0.5,1,1.5,2,2.5,3], colors='black')
    plt.setp(contours.collections , linewidth=1)

    contours2 = plt.contour(col, col, data.values, [1], colors='red')
    plt.setp(contours2.collections , linewidth=1.5)
    label_loc = [(10**1,10**1.3),(10**0.77,10**0.8),(10**0.63,10**0.53),(10**0.6,10**0.3),(10**0.4,10**(-.0)),(10**(-.0),10**(-.4))]
    levels = [0.5,1,1.5,2,2.5,3]
    for indx in range(len(label_loc)):
        word_x =label_loc[indx][0]
        word_y =label_loc[indx][1]
        if levels[indx] == 1:
            plt.text(word_x, word_y,  str(round(levels[indx],1)),color = 'red')
        else:
            plt.text(word_x, word_y, str(round(levels[indx], 1)), color='black')

    orig_cmap = matplotlib.cm.RdGy
    shrunk_cmap = shiftedColorMap(orig_cmap, start=0, midpoint=1, stop=3.3, name='shrunk')

    im = ax.imshow(data.values,  origin='lower', extent=[0.1, 100, 0.1, 100],
               cmap=shrunk_cmap, alpha=0.5)
    # plt.xlim([col[0],col[-1]])
    # plt.ylim([col2[0], col2[-1]])
    plt.yscale('log')
    plt.xscale('log')
    plt.yticks(fontsize=font_size)
    plt.yticks([0.1,1,10,100],[r'$10^1$',r'$10^0$',r'$10^{-1}$',r'$10^{-2}$'], fontsize=font_size)
    plt.xlabel('Scale of ' + r'$\mu$', fontsize=font_size)
    plt.ylabel('Scale of ' + r'$\beta_I$', fontsize=font_size)
    #
    plt.scatter([1],[1], marker = 'X',s = 100,color = 'black')
    plt.text(0.92,0.7, 'Status quo', color='black')
    #


    axins1 = inset_axes(ax,
                        width="5%",  # width = 50% of parent_bbox width
                        height="70%",  # height : 5%
                        bbox_to_anchor=(1.05,0.2,.7,.7), bbox_transform=ax.transAxes,
                        loc = 'upper left')

    fig.colorbar(im, cax=axins1, orientation="vertical")
    word_x = 0
    word_y = -1.7
    plt.text(word_x, word_y, 'Equivalent ' + r'$R_0$', color='black',rotation='vertical',fontsize = font_size)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/impact_beta_mu.eps', dpi = 200)

def plot_beta_impact(data, save_fig=1):
    col = np.linspace(0.1, 100, num=1000)


    data.columns = col
    data.index = col
    beta_data = data.loc[:,1] # mu = 1
    font_size = 16
    matplotlib.rcParams['font.size'] = font_size - 2
    fig, ax = plt.subplots(1, 1, figsize=[7, 3.5])
    x = list(col)
    y = beta_data.values
    colors = sns.color_palette('muted')
    plt.plot(x,y, color = colors[3])
    # plt.gca().invert_xaxis()

    plt.xlabel('Scale of ' + r'$\beta_I$', fontsize=font_size)
    plt.ylabel('Equivalent ' + r'$R_0$', fontsize=font_size)
    plt.xscale('log')
    plt.xticks([0.1, 1, 10, 100], [r'$10^1$', r'$10^0$', r'$10^{-1}$', r'$10^{-2}$'], fontsize=font_size)
    plt.scatter(1,beta_data.loc[beta_data.index == 1], marker = 'X',s = 100,color = 'black')
    plt.text(1, beta_data.loc[beta_data.index == 1] + 0.15, 'Status quo', color='black')
    plt.xlim([0.1, 100])
    plt.ylim([0, 3.5])
    plt.tight_layout()

    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/impact_beta.eps', dpi = 200)

def plot_mu_impact(data, save_fig=1):
    col = np.linspace(0.1, 100, num=1000)
    # col = np.log10(col)
    # # col = np.log10(col)

    data.columns = col
    data.index =col
    # data = data.rename(columns = {1:'reference'})
    beta_data = data.loc[1,:] # beta = 1
    font_size = 16
    matplotlib.rcParams['font.size'] = font_size - 2
    fig, ax = plt.subplots(1, 1, figsize=[7, 3.5])
    x = list(beta_data.index)
    y = beta_data.values
    colors = sns.color_palette('muted')
    plt.plot(x,y, color = colors[3])
    # plt.gca().invert_xaxis()

    plt.xlabel('Scale of ' + r'$\mu$', fontsize=font_size)
    plt.ylabel('Equivalent ' + r'$R_0$', fontsize=font_size)

    plt.scatter(1,beta_data.loc[beta_data.index == 1], marker = 'X',s = 100,color = 'black')
    plt.text(1, beta_data.loc[beta_data.index == 1]+0.1, 'Status quo', color='black')

    plt.ylim([0, 3.5])
    plt.xscale('log')
    plt.xlim([0.1, 100])
    plt.tight_layout()

    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/impact_mu.eps', dpi = 200)

if __name__ == '__main__':
    data = pd.read_csv('../data/impact_of_beta_E_mu.csv', header=None)
    plot_mu_impact(data, save_fig=1)
    plot_beta_impact(data, save_fig=1)
    plot_contour(data, save_fig=1)
