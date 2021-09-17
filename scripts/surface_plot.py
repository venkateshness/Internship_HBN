import nilearn
from nilearn import datasets
from nilearn import plotting

from surfplot import Plot

import matplotlib.pyplot as plt

import brainspace.mesh
fsaverage = datasets.fetch_surf_fsaverage()

mesh = brainspace.mesh.mesh_io.read_surface('brainnotation/tpl-fsaverage_den-10k_hemi-L_pial.surf.gii')
mesh2 = brainspace.mesh.mesh_io.read_surface('brainnotation/tpl-fsaverage_den-10k_hemi-R_pial.surf.gii')

def plot(data,title,cmap,view,c_range):
    p = Plot(mesh,mesh2, zoom=1.2, views=view)

    p.add_layer(data)
    fig = p.build()
    plt.title(title)
    return fig.show()