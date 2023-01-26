#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

total_no_of_events = '30_events'
roi = [0, 3, 27, 159]
label = ['V1', 'Pri Motor', 'STG', 'VentroMVisual']

sdi = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/SDI/{total_no_of_events}.npz')
sdi_f = sdi['alpha']

for n in range(len(roi)):
    # sns.heatmap((sdi_f[:,:,n]<0).T *1)
    # plt.xlabel('events')
    # plt.ylabel('subjects')
    # plt.title(f'PO / {label[n]}')
    # plt.show()

    sns.heatmap((sdi_f[:,:,n]<0).T *-1 + (sdi_f[:,:,n]>0).T *1, cmap = 'seismic')
    plt.xlabel('events')
    plt.ylabel('subjects')
    plt.title(f'PO / {label[n]}')
    plt.show()

# %%
sns.heatmap((sdi_f[:,:,87]<0).T *-1 + (sdi_f[:,:,87]>0).T *1, cmap = 'seismic')
# %%
final_data = ((sdi_f[:,:,87]<0).T *-1 + (sdi_f[:,:,87]>0).T *1)
final_data
# %%

syn1 = np.random.randint(5, size=(750,3))
import pandas as pd

df.columns = ['one', 'two', 'three']

df.columns
# %%
sns.jointplot(df, x ="one", y ="two")
# %%
import plotly.express as px
df = px.data.tips()

fig = px.density_heatmap(df, x="one", y="two", marginal_x="histogram", marginal_y="histogram")
fig.show()
# %%
import itertools

one = np.arange(25)
two = np.arange(30)

c = list(itertools.product(one, two))
# %%

one = [i for i, j in c]
two = [j for i, j in c]
three = final_data.flatten()

df = pd.DataFrame(one, two).reset_index()
df.columns = [ 'events', 'subjs']
df['SDI'] = three

# %%
fig = px.density_heatmap(df, x="events", y="subjs",z ='SDI', nbinsx = 30, nbinsy=25, histnorm = 'density')
fig.add_annotation(dict(font=dict(color='red',size=15),
                                        x=0,
                                        y=-0.12,
                                        showarrow=False,
                                        text=f"(sub) decoupling count = {np.sum((final_data==1)*final_data, axis= 1)}",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
fig.add_annotation(dict(font=dict(color='red',size=15),
x=0,
y=-0.22,
showarrow=False,
text=f"coupling = {np.sum((final_data==-1)*final_data, axis= 1)}",
textangle=0,
xanchor='left',
xref="paper",
yref="paper"))

fig.show()

# %%
np.sum( (final_data==1)*final_data, axis=0)

# %%
