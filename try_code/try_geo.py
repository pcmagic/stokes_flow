# coding=utf-8
import sys

from src import stokes_flow as sf
from src.geo import *
import numpy as np
from petsc4py import PETSc

# helix1 = supHelix()
# helix2 = helix1.create_n(R=0.8, B=0.1, r=0.1, n_node=500, n_c=1, eh=0.5)
# cbd_geo = geo()
# cbd_geo.combine(geo_list=[helix1, helix2, ])
# cbd_geo.show_nodes(linestyle='-')
# PETSc.Sys.Print(helix1.get_nodes_x().max(), helix1.get_nodes_x().min())
# PETSc.Sys.Print(helix1.get_nodes_y().max(), helix1.get_nodes_y().min())
# PETSc.Sys.Print(helix1.get_nodes_z().max(), helix1.get_nodes_z().min())
# # PETSc.Sys.Print(helix1.get_n_nodes())

# deltalength = 0.1
# deltatheta = np.pi/30
# length = 5
# radius = 1
# epsilon = 2
# tgeo = tunnel_geo()
# fgeo = tgeo.create_deltatheta(deltatheta, radius, length, epsilon, True, factor=10)
# tgeo.show_nodes()
# fgeo.show_nodes()
# cbd_geo = geo()
# cbd_geo.combine(geo_list=[tgeo, fgeo, ])
# cbd_geo.show_nodes(linestyle='-')

# deltalength = 0.1
# deltatheta = np.pi/6
# radius = 0.1
# epsilon = -1
# R = 0.6
# B = 0.1
# n_c = 1
# tgeo = supHelix()
# fgeo = tgeo.create_deltatheta(deltatheta, radius, R, B, n_c, epsilon, True, factor=0.5)
# # tgeo.show_nodes()
# # fgeo.show_nodes()
# cbd_geo = geo()
# cbd_geo.combine(geo_list=[tgeo, fgeo, ])
# cbd_geo.show_nodes(linestyle='-')

sphere0 = ellipse_geo()
sphere0.create_delta(0.07, 1, 0.5)
print(sphere0.get_n_nodes())
# sphere0.show_nodes()
