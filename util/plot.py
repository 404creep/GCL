import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import manifold
from sklearn.preprocessing import normalize
import seaborn as sns
from math import pi
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ue = open('user', 'rb')
user = pickle.load(ue)
udx = np.random.choice(len(user), 2000)
embs = ['user_lgcn.emb', 'user_sgl.emb', 'user_simgcl.emb']
models = ['LightGCN','SGL-ED','SimGCL']
data = {}


for emb,name in zip(embs,models):
    ue = open(emb, 'rb')
    user_emb = pickle.load(ue)
    selected_user_emb = user_emb[udx]  # np.concatenate([item_emb[idx],user_emb[udx]],axis=0)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    user_emb_2d = tsne.fit_transform(selected_user_emb)
    user_emb_2d = normalize(user_emb_2d, axis=1, norm='l2')
    data[name]=user_emb_2d


f, axs = plt.subplots(2, len(embs), figsize=(12,3.5),gridspec_kw={'height_ratios': [3, 1]})
kwargs = {'levels': np.arange(0, 5.5, 0.5)}
for i,name in enumerate(models):
    sns.kdeplot(data=data[name], bw=0.05, shade=True, cmap="GnBu", legend=True, ax=axs[0][i], **kwargs)
    axs[0][i].set_title(name, fontsize=9, fontweight="bold")
    x = [p[0] for p in data[name]]
    y = [p[1] for p in data[name]]
    angles = np.arctan2(y, x)
    sns.kdeplot(data=angles, bw=0.15, shade=True, legend=True, ax=axs[1][i], color='green')


for ax in axs[0]:
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.patch.set_facecolor('white')
    ax.collections[0].set_alpha(0)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Features', fontsize=9)
axs[0][0].set_ylabel('Features', fontsize=9)

for ax in axs[1]:
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('Angles', fontsize=9)
    ax.set_ylim(0, 0.5)
    ax.set_xlim(-pi, pi)
axs[1][0].set_ylabel('Density', fontsize=9)

plt.show()