import numpy as np 
import pandas as pd

from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA 

import matplotlib
import matplotlib.pyplot as plt 

import seaborn as sns 

import pdb 


def visualize(feats, labels):
   
    feats = np.moveaxis(feats,[0,1,2,3],[0,3,1,2])
    feats = np.reshape(feats, (-1,256))
    labels = np.reshape(labels, (-1,1))
    

    indices_ = np.where(labels[:,0] != 255)
    
    labels = labels[indices_[0],:]
    feats = feats[indices_[0],:]
    # pca = PCA(n_components = 10)
    # feats_pca = pca.fit_transform(feats)

    feats_tsne = TSNE(n_components=2, perplexity = 30, n_jobs = -1).fit_transform(feats)

    tsne_data = np.hstack((feats_tsne, labels))
    tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

    plt.figure(figsize=(16,10))
    sns.set_context('notebook', font_scale=1.1)
    sns.set_style("ticks")
    
    # sns.lmplot(x='Dim_1', y = 'Dim_2', data = tsne_df, fit_reg=False, legend=True, hue='label', palette=sns.color_palette("hls", 19), scatter_kws={"s":5, "alpha":0.3})
    sns.scatterplot( x="Dim_1", y="Dim_2", hue="label", 
    palette=sns.color_palette("hls", 19), data=tsne_df, legend="full",
    alpha=0.3
)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
 
    # plt.show()
    plt.savefig('plots/test_cos.png')
    plt.close()
    exit(0)
    



