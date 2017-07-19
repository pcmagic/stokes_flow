from StokesFlowMethod import *

node1 = np.array(((0,0,0), (1,2,3)), order='F')
node2 = np.array(((-1,-2,-3), (4,5,6)), order='F')
# node1 = np.array((0,0,0))
# node2 = np.array((1,2,3))
u1 = np.zeros(node1.size)
obj1 = sf.point_source_dipoleObj()
kwargs = {'pf_nodes': node1}
obj1.set_data(node1, node2, u1, **kwargs)

kwargs = {'ps_ds_para': 1}

M_petsc = point_source_dipole_matrix_3d_petsc(obj1, obj1, **kwargs)
M = M_petsc.getDenseArray().copy()
pass
