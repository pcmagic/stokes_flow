class stokesletsPreconditionProblem(stokesletsProblem, preconditionProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # for precondition problem
        self._inv_diag_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)  # diagonal elements of M matrix
        # for stokeslets problem
        self._stokeslets_post = np.array(kwargs['stokeslets_post']).reshape(1, 3)
        stokeslets_f = kwargs['stokeslets_f']
        self.set_stokeslets_f(stokeslets_f)


class stokesletsPreconditionObj(stokesletsObj, preconditionObj):
    def nothing(self):
        pass


class surf_forceObj(stokesFlowObj):
    def __init__(self, filename: str = '..'):
        super().__init__(filename)
        self._norm = np.zeros([0])  # information about normal vector at each point.

    def import_mat(self, filename: str):
        """
        Import geometries from Matlab file.

        :type filename: str
        :param filename: name of mat file containing object information
        """
        mat_contents = sio.loadmat(filename)
        velocity = mat_contents['U'].astype(np.float)
        origin = mat_contents['origin'].astype(np.float)
        f_nodes = mat_contents['f_nodes'].astype(np.float)
        u_nodes = mat_contents['u_nodes'].astype(np.float)
        norm = mat_contents['norm'].astype(np.float)
        para = {
            'norm':   norm,
            'origin': origin
        }
        self.set_data(f_nodes, u_nodes, velocity, **para)

    def import_nodes(self, filename: str):
        """
        Import geometries from Matlab file.

        :type filename: str
        :param filename: name of mat file containing object information
        """
        mat_contents = sio.loadmat(filename)
        origin = mat_contents['origin'].astype(np.float)
        f_nodes = mat_contents['f_nodes'].astype(np.float)
        u_nodes = mat_contents['u_nodes'].astype(np.float)
        norm = mat_contents['norm'].astype(np.float)
        para = {
            'norm':   norm,
            'origin': origin
        }
        self.set_nodes(f_nodes, u_nodes, **para)

    def set_nodes(self,
                  f_geo: geo,
                  u_geo: geo,
                  **kwargs):
        need_args = []
        for key in need_args:
            if not key in kwargs:
                err_msg = 'information about ' + key + \
                          ' is nesscery for surface force method. '
                raise ValueError(err_msg)

        args = {'origin': np.array([0, 0, 0])}
        for key, value in args.items():
            if not key in kwargs:
                kwargs[key] = args[key]

        super().set_nodes(f_geo, u_geo, **kwargs)
        self._norm = kwargs['norm']
        self._type = 'surface force obj'

    def get_norm(self):
        return self._norm


class pointSourceObj(stokesFlowObj):
    def __init__(self, filename: str = '..'):
        super().__init__(filename)
        self._n_unknown = 4

    def set_nodes(self,
                  f_geo: geo,
                  u_geo: geo,
                  **kwargs):
        super().set_nodes(f_geo, u_geo, **kwargs)
        self._type = 'point source obj'

    def vtk(self, filename):
        if str(self) == '...':
            return

        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            force_x = self._force[0::self._n_unknown].copy()
            force_y = self._force[1::self._n_unknown].copy()
            force_z = self._force[2::self._n_unknown].copy()
            pointSource = self._force[3::self._n_unknown].copy()
            velocity_x = self._re_velocity[0::3].copy()
            velocity_y = self._re_velocity[1::3].copy()
            velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])

            f_filename = filename + '_' + str(self) + '_force'
            pointsToVTK(f_filename, self.get_f_nodes()[:, 0], self.get_f_nodes()[:, 1], self.get_f_nodes()[:, 2],
                        data={"force":        (force_x, force_y, force_z),
                              "point_source": pointSource})
            u_filename = filename + '_' + str(self) + '_velocity'
            pointsToVTK(u_filename, self.get_u_nodes()[:, 0], self.get_u_nodes()[:, 1], self.get_u_nodes()[:, 2],
                        data={"velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z), })

            del force_x, force_y, force_z, \
                velocity_x, velocity_y, velocity_z, \
                velocity_err_x, velocity_err_y, velocity_err_z


class pointSourceProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_unknown = 4

    def vtk_force(self, filename):
        self.check_finish_solve()
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()

        if rank == 0:
            force_x = self._force[0::self._n_unknown].copy()
            force_y = self._force[1::self._n_unknown].copy()
            force_z = self._force[2::self._n_unknown].copy()
            pointSource = self._force[3::self._n_unknown].copy()
            velocity_x = self._re_velocity[0::3].copy()
            velocity_y = self._re_velocity[1::3].copy()
            velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])
            nodes = np.ones([self._f_node_index_list[-1], 3], order='F')
            for i, obj in enumerate(self._obj_list):
                nodes[self._f_node_index_list[i]:self._f_node_index_list[i + 1], :] = obj.get_f_nodes()
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"force":        (force_x, force_y, force_z),
                              "velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z),
                              "point_source": pointSource})
            del force_x, force_y, force_z, \
                velocity_x, velocity_y, velocity_z, \
                velocity_err_x, velocity_err_y, velocity_err_z, \
                nodes, pointSource


class point_source_dipoleObj(stokesFlowObj):
    def __init__(self, filename: str = '..'):
        super().__init__(filename)
        self._n_unknown = 6
        self._pf_geo = geo()
        self._pf_velocity = np.array(0)

    def set_nodes(self,
                  f_geo: geo,
                  u_geo: geo,
                  **kwargs):
        super().set_nodes(f_geo, u_geo, **kwargs)
        self._pf_geo = kwargs['pf_geo']
        self._type = 'dipole obj'

    def set_velocity(self,
                     velocity: np.array,
                     **kwargs):
        need_args = []
        for key in need_args:
            if not key in kwargs:
                err_msg = 'information about ' + key + \
                          ' is nesscery for surface force method. '
                raise ValueError(err_msg)

        args = {'pf_velocity': np.zeros(velocity.shape)}
        for key, value in args.items():
            if not key in kwargs:
                kwargs[key] = args[key]

        self._velocity = velocity.reshape(velocity.size)
        self._pf_velocity = kwargs['pf_velocity']

    def vtk(self, filename):
        if str(self) == '...':
            return

        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            num_unknown = self._n_unknown
            force_x = self._force[0::num_unknown].copy()
            force_y = self._force[1::num_unknown].copy()
            force_z = self._force[2::num_unknown].copy()
            dipole_x = self._force[3::num_unknown].copy()
            dipole_y = self._force[4::num_unknown].copy()
            dipole_z = self._force[5::num_unknown].copy()
            velocity_x = self._re_velocity[0::3].copy()
            velocity_y = self._re_velocity[1::3].copy()
            velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self._re_velocity[0::3] - self.get_velocity()[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self.get_velocity()[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self.get_velocity()[2::3])

            f_filename = filename + '_' + str(self) + '_force'
            pointsToVTK(f_filename, self.get_f_nodes()[:, 0], self.get_f_nodes()[:, 1], self.get_f_nodes()[:, 2],
                        data={"force":  (force_x, force_y, force_z),
                              "dipole": (dipole_x, dipole_y, dipole_z), })
            u_filename = filename + '_' + str(self) + '_velocity'
            pointsToVTK(u_filename, self.get_u_nodes()[:, 0], self.get_u_nodes()[:, 1], self.get_u_nodes()[:, 2],
                        data={"velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z), })
        return True

    def get_pf_geo(self):
        return self._pf_geo

    def get_pf_velocity(self):
        return self._pf_velocity


class point_source_dipoleProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_unknown = 6
        self._ini_problem = []

    def vtk_force(self, filename):
        self.check_finish_solve()
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()

        if rank == 0:
            num_unknown = self._n_unknown
            force_x = self._force[0::num_unknown].copy()
            force_y = self._force[1::num_unknown].copy()
            force_z = self._force[2::num_unknown].copy()
            dipole_x = self._force[3::num_unknown].copy()
            dipole_y = self._force[4::num_unknown].copy()
            dipole_z = self._force[5::num_unknown].copy()
            velocity_x = self._re_velocity[0::3].copy()
            velocity_y = self._re_velocity[1::3].copy()
            velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])
            nodes = np.ones([self._f_node_index_list[-1], 3], order='F')
            for i, obj in enumerate(self._obj_list):
                nodes[self._f_node_index_list[i]:self._f_node_index_list[i + 1], :] = obj.get_f_nodes()
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"force":        (force_x, force_y, force_z),
                              "velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z),
                              "dipole":       (dipole_x, dipole_y, dipole_z), })

    def ini_guess(self):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 1:
            print('%s: get ini guess for ps_ds method.' % str(self))

        ini_problem = self.ini_problem()
        ini_problem.create_matrix()
        residualNorm = ini_problem.solve()
        pf_force = ini_problem.get_force()
        temp0 = np.array(pf_force).reshape((-1, 3))
        ini_guess = np.hstack((temp0, np.zeros(temp0.shape))).flatten()

        return ini_guess, residualNorm, ini_problem

    def ini_problem(self):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 1:
            print('%s: generate ini guess problem.' % str(self))

        problem_arguments = self._kwargs.copy()
        problem_arguments['matrix_method'] = 'pf'
        problem_arguments['name'] = 'ini_' + problem_arguments['name']
        problem_arguments['fileHeadle'] += '_ini'
        ini_problem = stokesFlowProblem(**problem_arguments)
        for obj0 in self.get_obj_list():
            obj1 = stokesFlowObj()
            para = {'name': obj0.get_name()}
            obj1.set_data(obj0.get_f_geo(), obj0.get_pf_geo(), obj0.get_pf_velocity(), **para)
            ini_problem.add_obj(obj1)

        self._ini_problem = ini_problem
        return ini_problem

    # def solve(self, ini_guess=None, Tolerances={}):
    #     if ini_guess is None:
    #         ini_guess = np.zeros(self.get_n_f_node() * self.get_n_unknown())
    #
    #     return super().solve(ini_guess=ini_guess, Tolerances=Tolerances)

    def get_ini_problem(self):
        return self._ini_problem


class point_force_dipoleObj(stokesFlowObj):
    def __init__(self, filename: str = '..'):
        super().__init__(filename)
        self._n_unknown = 12

    def set_nodes(self,
                  f_geo: geo,
                  u_geo: geo,
                  **kwargs):
        super().set_nodes(f_geo, u_geo, **kwargs)
        self._type = 'dipole obj'

    def vtk(self, filename):
        if str(self) == '...':
            return

        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            num_unknown = self._n_unknown
            force_x = self._force[0::num_unknown].copy()
            force_y = self._force[1::num_unknown].copy()
            force_z = self._force[2::num_unknown].copy()
            dipole_1 = self._force[3::num_unknown].copy()
            dipole_2 = self._force[4::num_unknown].copy()
            dipole_3 = self._force[5::num_unknown].copy()
            dipole_4 = self._force[6::num_unknown].copy()
            dipole_5 = self._force[7::num_unknown].copy()
            dipole_6 = self._force[8::num_unknown].copy()
            dipole_7 = self._force[9::num_unknown].copy()
            dipole_8 = self._force[10::num_unknown].copy()
            dipole_9 = self._force[11::num_unknown].copy()
            velocity_x = self._re_velocity[0::3].copy()
            velocity_y = self._re_velocity[1::3].copy()
            velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self._re_velocity[0::3] - self.get_velocity()[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self.get_velocity()[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self.get_velocity()[2::3])

            f_filename = filename + '_' + str(self) + '_force'
            pointsToVTK(f_filename, self.get_f_nodes()[:, 0], self.get_f_nodes()[:, 1], self.get_f_nodes()[:, 2],
                        data={"force":   (force_x, force_y, force_z),
                              "dipole1": (dipole_1, dipole_2, dipole_3),
                              "dipole2": (dipole_4, dipole_5, dipole_6),
                              "dipole3": (dipole_7, dipole_8, dipole_9)})
            u_filename = filename + '_' + str(self) + '_velocity'
            pointsToVTK(u_filename, self.get_u_nodes()[:, 0], self.get_u_nodes()[:, 1], self.get_u_nodes()[:, 2],
                        data={"velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z), })
        return True


class point_force_dipoleProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_unknown = 12

    def vtk_force(self, filename):
        self.check_finish_solve()
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()

        if rank == 0:
            num_unknown = self._n_unknown
            force_x = self._force[0::num_unknown].copy()
            force_y = self._force[1::num_unknown].copy()
            force_z = self._force[2::num_unknown].copy()
            dipole_1 = self._force[3::num_unknown].copy()
            dipole_2 = self._force[4::num_unknown].copy()
            dipole_3 = self._force[5::num_unknown].copy()
            dipole_4 = self._force[6::num_unknown].copy()
            dipole_5 = self._force[7::num_unknown].copy()
            dipole_6 = self._force[8::num_unknown].copy()
            dipole_7 = self._force[9::num_unknown].copy()
            dipole_8 = self._force[10::num_unknown].copy()
            dipole_9 = self._force[11::num_unknown].copy()
            velocity_x = self._re_velocity[0::3].copy()
            velocity_y = self._re_velocity[1::3].copy()
            velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])
            nodes = np.ones([self._f_node_index_list[-1], 3], order='F')
            for i, obj in enumerate(self._obj_list):
                nodes[self._f_node_index_list[i]:self._f_node_index_list[i + 1], :] = obj.get_f_nodes()
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"force":        (force_x, force_y, force_z),
                              "velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z),
                              "dipole":       (
                                  dipole_1, dipole_2, dipole_3, dipole_4, dipole_5, dipole_6, dipole_7, dipole_8,
                                  dipole_9), })


class preconditionObj(stokesFlowObj):
    def __init__(self, filename: str = '..'):
        super().__init__(filename)
        self._inv_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)

    def create_inv_diag_matrix(self):
        t0 = time()
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        problem = self.get_problem()
        # assert isinstance(problem, stokesFlowProblem)
        matrix_method = problem.get_matrix_method()
        kwargs = problem.get_kwargs()
        temp_m_petsc = matrix_method(self, self, **kwargs)

        I_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        temp_m_petsc.convert(mat_type='dense', out=I_petsc)
        I_petsc.zeroEntries()
        I_petsc.shift(alpha=1)
        temp_m_petsc.convert(mat_type='dense', out=self._inv_petsc)
        rperm, cperm = temp_m_petsc.getOrdering(ord_type='MATORDERINGNATURAL')
        temp_m_petsc.factorLU(rperm, cperm)
        # temp_m_petsc.factorCholesky(rperm)
        temp_m_petsc.matSolve(I_petsc, self._inv_petsc)
        temp_m_petsc.destroy()
        I_petsc.destroy()

        t1 = time()
        PETSc.Sys.Print('%s: solve inverse matrix: %fs.' % (str(self), (t1 - t0)))
        return True

    def get_inv_petsc(self):
        return self._inv_petsc

    def set_problem(self, problem: 'stokesFlowProblem'):
        super().set_problem(problem)

        if not self.get_inv_petsc().isAssembled():
            self.create_inv_diag_matrix()
        return True


class preconditionProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._inv_diag_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)  # diagonal elements of M matrix

    def create_inv_diag_matrix(self):
        """
        calculate self-to-self elements for the M matrix.
        :return:
        """
        if not self._inv_diag_petsc.isAssembled():
            self._inv_diag_petsc.setSizes(((None, self._u_index_list[-1]), (None, self._f_index_list[-1])))
            self._inv_diag_petsc.setType('dense')
            self._inv_diag_petsc.setFromOptions()
            self._inv_diag_petsc.setUp()

        for i, obj1 in enumerate(self._obj_list):
            inv_petsc = obj1.get_inv_petsc()
            err_msg = 'create inv matrix of %s first. ' % str(obj1)
            assert inv_petsc.isAssembled(), err_msg

            velocity_index_begin = self._f_index_list[i]
            force_index_begin = self._f_index_list[i]
            force_index_end = self._f_index_list[i + 1]
            temp_m_start, temp_m_end = inv_petsc.getOwnershipRange()

            temp_inv_petsc = inv_petsc.getDenseArray()
            for k in range(temp_m_start, temp_m_end):
                self._inv_diag_petsc.setValues(velocity_index_begin + k,
                                               np.arange(force_index_begin, force_index_end, dtype='int32'),
                                               temp_inv_petsc[k - temp_m_start, :])
        self._inv_diag_petsc.assemble()

        return True

    def create_matrix(self):
        self.create_inv_diag_matrix()
        super().create_matrix()
        return True

    def solve(self, ini_guess=None, Tolerances={}):
        err_msg = 'create the inverse of main-obj diagonal matrix first. '
        assert self._inv_diag_petsc.isAssembled(), err_msg
        assert 1 == 2, 'DO NOT WORK, self._velocity property is changing. '

        old_M_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        self._M_petsc.convert(mat_type='dense', out=old_M_petsc)
        self._inv_diag_petsc.matMult(old_M_petsc, self._M_petsc)

        old_velocity = self._velocity.copy()
        inv_diag = self._inv_diag_petsc.getDenseArray()
        self._velocity = inv_diag.dot(old_velocity)

        super().solve(ini_guess=ini_guess, Tolerances=Tolerances)
        self._M_petsc = old_M_petsc
        self._velocity = old_velocity

        return True


class stokesletsProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stokeslets_post = np.array(kwargs['stokeslets_post']).reshape(1, 3)
        stokeslets_f = kwargs['stokeslets_f']
        self.set_stokeslets_f(stokeslets_f)

    def get_stokeslets_post(self):
        return self._stokeslets_post

    def set_stokeslets_post(self, stokeslets_post):
        errmsg = 'stokeslets_post (stokeslets postion) is a np.ndarray obj in shape (1, 3)'
        assert isinstance(stokeslets_post, np.ndarray), errmsg
        assert stokeslets_post.shape == (1, 3), errmsg
        self._stokeslets_post = stokeslets_post
        return True

    def get_stokeslets_f(self):
        return self._stokeslets_f

    def set_stokeslets_f(self, stokeslets_f):
        errmsg = 'stokeslets_f (stokeslets force) is a np.ndarray obj in shape (3, )'
        assert isinstance(stokeslets_f, np.ndarray), errmsg
        assert stokeslets_f.shape == (3,), errmsg
        self._stokeslets_f = stokeslets_f
        self._stokeslets_f_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self._stokeslets_f_petsc.setSizes(stokeslets_f.size)
        self._stokeslets_f_petsc.setFromOptions()
        self._stokeslets_f_petsc.setUp()
        self._stokeslets_f_petsc[:] = stokeslets_f[:]
        self._stokeslets_f_petsc.assemble()
        return True

    def unpickmyself(self):
        super().unpickmyself()
        stokeslets_f = self.get_stokeslets_f()
        self.set_stokeslets_f(stokeslets_f)
        return True

    def vtk_velocity(self, filename: str):
        self.check_finish_solve()
        field_range, n_grid = self.check_vtk_velocity()
        region_type = self._kwargs['region_type']

        if not self._M_destroyed:
            self._M_petsc.destroy()
            self._M_destroyed = True
        myregion = region()
        full_region_x, full_region_y, full_region_z = myregion.type[region_type](field_range, n_grid)

        # solve velocity at cell center.
        n_para = 3 * n_grid[1] * n_grid[2]
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()

        # to handle big problem, solve velocity field at every splice along x axis.
        # u_total = u1 + u2. satisfying boundary conditions.
        # u1: cancelled background flow at boundary.
        # u2: background flow due to stokeslets.
        if rank == 0:
            u_x = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u_y = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u_z = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u1_x = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u1_y = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u1_z = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u2_x = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u2_y = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u2_z = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
        else:
            u_x = None
            u_y = None
            u_z = None
        m1_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        m1_petsc.setSizes(((None, n_para), (None, self._f_index_list[-1])))
        m1_petsc.setType('dense')
        m1_petsc.setFromOptions()
        m1_petsc.setUp()
        obj1 = stokesFlowObj()
        geo_stokeslets = geo()
        geo_stokeslets.set_nodes(self.get_stokeslets_post(), deltalength=0, resetVelocity=True)
        obj_stokeslets = stokesFlowObj()
        obj_stokeslets.set_data(geo_stokeslets, geo_stokeslets)
        for i0 in range(full_region_x.shape[0]):
            # u1: cancelled background flow to satisfy boundary conditions.
            temp_x = full_region_x[i0]
            temp_y = full_region_y[i0]
            temp_z = full_region_z[i0]
            temp_nodes = np.c_[temp_x.ravel(), temp_y.ravel(), temp_z.ravel()]
            temp_geo = geo()
            temp_geo.set_nodes(temp_nodes, deltalength=0, resetVelocity=True)
            obj1.set_data(temp_geo, temp_geo)
            self.create_matrix_obj(obj1, m1_petsc)
            m1_petsc.assemble()
            u1_petsc = m1_petsc.createVecLeft()
            # u1_petsc.set(0)
            m1_petsc.mult(self._force_petsc, u1_petsc)
            u1 = self.vec_scatter(u1_petsc)
            # u2: add background flow due to stokeslets.
            from src.StokesFlowMethod import stokeslets_matrix_3d_petsc
            m2_petsc = stokeslets_matrix_3d_petsc(obj1, obj_stokeslets)
            u2_petsc = m2_petsc.createVecLeft()
            # u2_petsc.set(0)
            m2_petsc.mult(self._stokeslets_f_petsc, u2_petsc)
            u2 = self.vec_scatter(u2_petsc)
            m2_petsc.destroy()
            if rank == 0:
                u1_x[i0, :, :] = u1[0::3].reshape((n_grid[1], n_grid[2])).copy()
                u1_y[i0, :, :] = u1[1::3].reshape((n_grid[1], n_grid[2])).copy()
                u1_z[i0, :, :] = u1[2::3].reshape((n_grid[1], n_grid[2])).copy()
                u2_x[i0, :, :] = u2[0::3].reshape((n_grid[1], n_grid[2])).copy()
                u2_y[i0, :, :] = u2[1::3].reshape((n_grid[1], n_grid[2])).copy()
                u2_z[i0, :, :] = u2[2::3].reshape((n_grid[1], n_grid[2])).copy()
                u_x[i0, :, :] = u1_x[i0, :, :] + u2_x[i0, :, :]
                u_y[i0, :, :] = u1_y[i0, :, :] + u2_y[i0, :, :]
                u_z[i0, :, :] = u1_z[i0, :, :] + u2_z[i0, :, :]
            else:
                u_x = None
                u_y = None
                u_z = None

        m1_petsc.destroy()
        if rank == 0:
            # output data
            gridToVTK(filename, full_region_x, full_region_y, full_region_z,
                      pointData={"u1": (u1_x, u1_y, u1_z),
                                 "u2": (u2_x, u2_y, u2_z),
                                 "u":  (u_x, u_y, u_z)})
        return True

    def vtk_tetra(self,
                  filename: str,
                  bgeo: geo):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        self.check_finish_solve()

        bnodes = bgeo.get_nodes()
        belems, elemtype = bgeo.get_mesh()
        err_msg = 'mesh type is NOT tetrahedron. '
        assert elemtype == 'tetra', err_msg

        PETSc.Sys.Print('export to %s.vtk, element type is %s, contain %d nodes and %d elements. '
                        % (filename, elemtype, bnodes.shape[0], belems.shape[0]))

        n_para = bgeo.get_n_nodes() * 3
        # u1: cancelled background flow to satisfy boundary conditions.
        m1_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        m1_petsc.setSizes(((None, n_para), (None, self._f_index_list[-1])))
        m1_petsc.setType('dense')
        m1_petsc.setFromOptions()
        m1_petsc.setUp()
        obj1 = stokesFlowObj()
        obj1.set_data(bgeo, bgeo, np.zeros(bnodes.size))
        self.create_matrix_obj(obj1, m1_petsc)
        m1_petsc.assemble()
        u1_petsc = m1_petsc.createVecLeft()
        # u1_petsc.set(0)
        m1_petsc.mult(self._force_petsc, u1_petsc)
        u1 = self.vec_scatter(u1_petsc)
        # u2: add background flow due to stokeslets.
        from src.StokesFlowMethod import stokeslets_matrix_3d_petsc
        geo_stokeslets = geo()
        geo_stokeslets.set_nodes(self.get_stokeslets_post(), deltalength=0, resetVelocity=True)
        obj_stokeslets = stokesFlowObj()
        obj_stokeslets.set_data(geo_stokeslets, geo_stokeslets)
        m2_petsc = stokeslets_matrix_3d_petsc(obj1, obj_stokeslets)
        u2_petsc = m2_petsc.createVecLeft()
        # u2_petsc.set(0)
        m2_petsc.mult(self._stokeslets_f_petsc, u2_petsc)
        u2 = self.vec_scatter(u2_petsc)
        m1_petsc.destroy()
        m2_petsc.destroy()

        if rank == 0:
            u1 = np.array(u1).reshape(bnodes.shape)
            u2 = np.array(u2).reshape(bnodes.shape)
            u = u1 + u2
            vtk = VtkData(
                    UnstructuredGrid(bnodes,
                                     tetra=belems,
                                     ),
                    PointData(Vectors(u, name='u'),
                              Vectors(u1, name='u1'),
                              Vectors(u2, name='u2')),
                    str(self)
            )
            vtk.tofile(filename)
        return True


class stokesletsObj(stokesFlowObj):
    def __init__(self):
        super().__init__()
        self._stokeslets_post = np.array((0, 0, 0)).reshape(1, 3)
        self._stokeslets_f = np.array((0, 0, 0)).reshape(1, 3)
        self._stokeslets_f_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)

    def set_data(self,
                 f_geo: geo,
                 u_geo: geo,
                 name='...', **kwargs):
        super().set_data(f_geo, u_geo, name, **kwargs)
        self._stokeslets_post = np.array(kwargs['stokeslets_post']).reshape(1, 3)
        stokeslets_f = kwargs['stokeslets_f']
        self.set_stokeslets_f(stokeslets_f)

    def get_stokeslets_post(self):
        return self._stokeslets_post

    def set_stokeslets_post(self, stokeslets_post):
        errmsg = 'stokeslets_post (stokeslets postion) is a np.ndarray obj in shape (1, 3)'
        assert isinstance(stokeslets_post, np.ndarray), errmsg
        assert stokeslets_post.shape == (1, 3), errmsg
        self._stokeslets_post = stokeslets_post
        return True

    def get_stokeslets_f(self):
        return self._stokeslets_f

    def set_stokeslets_f(self, stokeslets_f):
        errmsg = 'stokeslets_f (stokeslets force) is a np.ndarray obj in shape (3, )'
        assert isinstance(stokeslets_f, np.ndarray), errmsg
        assert stokeslets_f.shape == (3,), errmsg
        self._stokeslets_f = stokeslets_f
        self._stokeslets_f_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self._stokeslets_f_petsc.setSizes(stokeslets_f.size)
        self._stokeslets_f_petsc.setFromOptions()
        self._stokeslets_f_petsc.setUp()
        self._stokeslets_f_petsc[:] = stokeslets_f[:]
        self._stokeslets_f_petsc.assemble()
        return True

    def unpickmyself(self):
        super().unpickmyself()
        stokeslets_f = self.get_stokeslets_f()
        self.set_stokeslets_f(stokeslets_f)
        return True

    def vtk(self, filename):
        if str(self) == '...':
            return
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()

        # u2: add background flow due to stokeslets.
        from src.StokesFlowMethod import stokeslets_matrix_3d_petsc
        geo_stokeslets = geo()
        geo_stokeslets.set_nodes(self.get_stokeslets_post())
        obj_stokeslets = stokesFlowObj()
        obj_stokeslets.set_data(geo_stokeslets, geo_stokeslets, np.zeros(self.get_stokeslets_post().size))
        m2_petsc = stokeslets_matrix_3d_petsc(self, obj_stokeslets)
        u2_petsc = m2_petsc.createVecLeft()
        # u2_petsc.set(0)
        m2_petsc.mult(self._stokeslets_f_petsc, u2_petsc)
        u2 = self.vec_scatter(u2_petsc)

        if rank == 0:
            force_x = self._force[0::self._n_unknown].copy()
            force_y = self._force[1::self._n_unknown].copy()
            force_z = self._force[2::self._n_unknown].copy()
            # u1: cancelled background flow to satisfy boundary conditions.
            re_velocity_x = self._re_velocity[0::3].copy()
            re_velocity_y = self._re_velocity[1::3].copy()
            re_velocity_z = self._re_velocity[2::3].copy()
            velocity_err_x = np.abs(self.get_velocity()[0::3].copy() - re_velocity_x)
            velocity_err_y = np.abs(self.get_velocity()[1::3].copy() - re_velocity_y)
            velocity_err_z = np.abs(self.get_velocity()[2::3].copy() - re_velocity_z)
            velocity_x = self.get_velocity()[0::3].copy()
            velocity_y = self.get_velocity()[1::3].copy()
            velocity_z = self.get_velocity()[2::3].copy()
            # u2: add background flow due to stokeslets.
            stokeslets_x = u2[0::3].copy()
            stokeslets_y = u2[1::3].copy()
            stokeslets_z = u2[2::3].copy()
            # u = u1 + u2
            u_x = velocity_x + stokeslets_x
            u_y = velocity_y + stokeslets_y
            u_z = velocity_z + stokeslets_z
            re_u_x = re_velocity_x + stokeslets_x
            re_u_y = re_velocity_y + stokeslets_y
            re_u_z = re_velocity_z + stokeslets_z

            f_filename = filename + '_' + str(self) + '_force'
            pointsToVTK(f_filename, self.get_f_nodes()[:, 0], self.get_f_nodes()[:, 1], self.get_f_nodes()[:, 2],
                        data={"force": (force_x, force_y, force_z), })
            u_filename = filename + '_' + str(self) + '_velocity'
            pointsToVTK(u_filename, self.get_u_nodes()[:, 0], self.get_u_nodes()[:, 1], self.get_u_nodes()[:, 2],
                        data={"re_u":  (re_u_x, re_u_y, re_u_z),
                              "re_u1": (re_velocity_x, re_velocity_y, re_velocity_z),
                              "re_u2": (stokeslets_x, stokeslets_y, stokeslets_z),
                              "u":     (u_x, u_y, u_z),
                              "u1":    (velocity_x, velocity_y, velocity_z),
                              "u2":    (stokeslets_x, stokeslets_y, stokeslets_z),
                              "u_err": (velocity_err_x, velocity_err_y, velocity_err_z), })

        return True
