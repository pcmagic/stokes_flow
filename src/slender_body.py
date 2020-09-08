from src import stokes_flow as sf
from src import SlenderBodyMethod as slbm
from src import StokesFlowMethod as sfm


class SlenderBodyProblem(sf.StokesFlowProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._slb_nonlocal_method = {
            'lightill_slb': slbm.SLB_matrix_nonlocal_petsc,
            'KRJ_slb':      slbm.SLB_matrix_nonlocal_petsc,
            'mod_KRJ_slb':  slbm.SLB_matrix_nonlocal_petsc,
        }
        self._slb_local_method = {
            'lightill_slb': slbm.Lightill_matrix_local_petsc,
            'KRJ_slb':      slbm.KRJ_matrix_local_petsc,
            'mod_KRJ_slb':  slbm.mod_KRJ_matrix_local_petsc,
        }
        self._check_args_dict['lightill_slb'] = sfm.check_point_force_matrix_3d_petsc
        self._check_args_dict['KRJ_slb'] = sfm.check_point_force_matrix_3d_petsc
        self._check_args_dict['mod_KRJ_slb'] = sfm.check_point_force_matrix_3d_petsc

    def _create_matrix_obj(self, obj1, m_petsc, INDEX='', *args):
        # obj1 contain velocity information, obj2 contain force information
        kwargs = self._kwargs
        n_obj = len(self.get_all_obj_list())
        for i0, obj2 in enumerate(self.get_all_obj_list()):
            kwargs['INDEX'] = ' %d/%d, ' % (i0 + 1, n_obj) + INDEX
            self._slb_nonlocal_method[obj2.get_matrix_method()](obj1, obj2, m_petsc, **kwargs)
        self._slb_local_method[obj1.get_matrix_method()](obj1, obj1, m_petsc, **kwargs)
        m_petsc.assemble()
        return True

# class SlenderBodyObj(sf.StokesFlowObj):
#     def move(self, displacement):
#         super().move(displacement)
#
#         return True

class StrainRateBaseForceFreeProblem(SlenderBodyProblem,
                                     sf.StrainRateBaseProblem,
                                     sf._GivenFlowForceFreeProblem):
    def _nothing(self):
        pass

problem_dic = {
    'lightill_slb': SlenderBodyProblem,
    'KRJ_slb':      SlenderBodyProblem,
    'mod_KRJ_slb':  SlenderBodyProblem,
}

obj_dic = {
    'lightill_slb': sf.StokesFlowObj,
    'KRJ_slb':      sf.StokesFlowObj,
    'mod_KRJ_slb':  sf.StokesFlowObj,
}

def check_matrix_method(matrix_method):
    keys = problem_dic.keys()
    err_msg = 'error matrix method. '
    assert matrix_method in keys, err_msg