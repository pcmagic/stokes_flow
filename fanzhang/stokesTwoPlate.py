import stokes_ as sk

'''
The commom package is already loaded in sk. 
np -- numpy; mp -- mpmath; plt -- matplotlib; 
sb -- seaborn; pd -- pandas
'''

class tank():
    ''' 
        we set the first plate as the plane of z=0.
        It's used for generating  field value table.
        
        (x_R, y_R, z_R)：the field position. Default value = (0, 0, 0)
        (x_F, y_F, z_F): the position of point force. In fact, we won't use it.
        Nx, Ny, Nz: the number of points. To be added...
        xlim, ylim, zlim: ragion of calculation. To be added...
        height: z_F, the height of force point.
        Height: the height of the second plate.
        u_tensor: the green function tensor. 
        U_tensor: table that contains u_tensor in every point.
        However, since the x-y plane is infinite large. we set (x_F, y_F) = (0, 0)
        So users only need to input z_F, or in other words, height.
    ''' 
    
    h0 = 1
    k = 2 * sk.mp.pi
    
    def __init__(self, Height):
        
        self.Height = Height
        #self.Nx, self.Ny, self.Nz = 1, 1, 1
        self.x_R = 0
        self.y_R = 0
        self.z_R = 0
        self.x_F = 0
        self.y_F = 0
        self.z_F = 0
        self.height = 0
        self.u_tensor = sk.np.zeros([3, 3]) 
        self.U_tensor = []
        self.far_integral = sk.far_integral
        self.near_integral = sk.near_integral 
		
    def set_locF(self, x_F, y_F, z_F):
        '''Set the location of force.'''
        self.x_F, self.y_F, self.z_F = x_F, y_F, z_F
        self.height = self.z_F
    
    def set_height(self, height):
        self.height = height
    	
    def get_ux(self, x, y, z, imax=4):
        '''Get Green function for ux'''
        para_1 = [x, y, z, self.height, self.Height, imax]
        para_2 = [y, x, z, self.height, self.Height, imax]

        self.u_tensor[0][0] = sk.u11(*para_1)
        self.u_tensor[0][1] = sk.u12(*para_1)
        self.u_tensor[0][2] = sk.u13(*para_1)
        
        return self.u_tensor[0]
    
    def get_uy(self, x, y, z, imax=4):
        '''Get Green function for uy'''
        para_1 = [x, y, z, self.height, self.Height, imax]
        para_2 = [y, x, z, self.height, self.Height, imax]

        self.u_tensor[1][0] = sk.u12(*para_1)
        self.u_tensor[1][1] = sk.u11(*para_2)
        self.u_tensor[1][2] = sk.u13(*para_2)
        
        return self.u_tensor[1]
    
    def get_uz(self, x, y, z, imax=4):
        '''Get Green function for uz'''
        para_1 = [x, y, z, self.height, self.Height, imax]
        para_2 = [y, x, z, self.height, self.Height, imax]

        self.u_tensor[2][0] = sk.u31(*para_1)
        self.u_tensor[2][1] = sk.u31(*para_2) 
        self.u_tensor[2][2] = sk.u33(*para_1)
        
        return self.u_tensor[2]
    
    def get_u(self, x, y, z, imax=4):
        '''Get Green function for one point.'''
        # Seperate xy plane for its symmetry.
        # para_0 for u11, u12, u21, u13, u31,u33.
        # para_1 for u22, u32, u32.

        para_1 = [x, y, z, self.height, self.Height, imax]
        para_2 = [y, x, z, self.height, self.Height, imax]

        self.u_tensor[0][0] = sk.u11(*para_1)
        self.u_tensor[0][1] = sk.u12(*para_1)
        self.u_tensor[1][0] = sk.u12(*para_1)
        self.u_tensor[0][2] = sk.u13(*para_1)
        self.u_tensor[2][0] = sk.u31(*para_1)


        self.u_tensor[1][1] = sk.u11(*para_2)
        self.u_tensor[1][2] = sk.u13(*para_2)
        self.u_tensor[2][1] = sk.u31(*para_2)
        
        self.u_tensor[2][2] = sk.u33(*para_1)

        return self.u_tensor.copy()
    
    def Ndiv(self, x, y, z, dr=1e-6):
        '''Return divgent of three Green functions.'''
        para_xu = [x+dr, y, z]
        para_xd = [x-dr, y, z]
        para_yu = [x, y+dr, z]
        para_yd = [x, y-dr, z]
        para_zu = [x, y, z+dr]
        para_zd = [x, y, z-dr]
        div_x = self.get_ux(*para_xu)/(2*dr) - self.get_ux(*para_xd)/(2*dr)
        div_y = self.get_uy(*para_yu)/(2*dr) - self.get_uy(*para_yd)/(2*dr)
        div_z = self.get_uz(*para_zu)/(2*dr) - self.get_uz(*para_zd)/(2*dr)
        
        return div_x + div_y + div_z

    def get_U(self, x_list, y_list, z_list, imax=4):
        '''Get the full space Green function.'''
        if (type(x_list) or type(y_list) or type(z_list)) is not sk.np.ndarray:
            raise TypeError("x, y, z should be array!")
        
        Uxy = sk.np.array([[[self.get_u(xx, yy, zz, imax) 
                          for xx in x_list] 
                          for yy in y_list]
                          for zz in z_list])
        return Uxy
	
    '''Added in 8.7, 2017. Input array, return full Green funciton.'''
    def get_ufunc(self, point_loc): 
        '''      
        First We get the coordinate of point relative to force,
        e.g, (x-x_F, y-y_F, z). (Not z-z_F!)
        When z > h, we use oscillation integral(near_integral) to 
        get the green funciton if z < 3h and normal integral(far-
        integral) else.
        For the z < h case, just replace the z and h to H-z and H-z
        under the integral.
        '''
        # func_list_u = [j0x2a1, j1xa1, j0a2, j1xa4_13, j1xa4_31]
        
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        r = sk.np.sqrt(x**2 + y**2)
        (z_, h_) = (z, self.height) if z > self.height else (self.Height - z, self.Height - self.height)
        paraw = [r, z, self.height, self.Height]
        parav = [r, z_, h_, self.Height]
        coeff = z - self.height
        far_integral = sk.far_integral
        near_integral = sk.near_integral
        
        if z_ < 3. * h_:
            j0sh = near_integral(sk.j0sh, *parav, m=0)
            j1xsh = near_integral(sk.j1xsh, *parav, m=1)
            j0xdsh = near_integral(sk.j0xdsh, *parav, m=0)
        else:
            imaxv = 9. * r / (z_ - h_)
            j0sh = far_integral(sk.j0sh, *paraw, imaxv, method='gauss-legendre')
            j1xsh = far_integral(sk.j1xsh, *paraw, imaxv, method='gauss-legendre')
            j0xdsh = far_integral(sk.j0xdsh, *paraw, imaxv, method='gauss-legendre')

        imaxw = 9. * r / (z + self.height)
        j0x2a1 = sk.far_integral(sk.j0x2a1, *paraw, imaxw, method='gauss-legendre')
        j1xa1 = far_integral(sk.j1xa1, *paraw, imaxw, method='gauss-legendre')
        j0a2 = far_integral(sk.j0a2, *paraw, imaxw, method='gauss-legendre')
        j1xa4_13 = far_integral(sk.j1xa4_13, *paraw, imaxw, method='gauss-legendre')
        j1xa4_31 = far_integral(sk.j1xa4_31, *paraw, imaxw, method='gauss-legendre')
        
        self.u_tensor[0][0] = (j0sh + x**2 / r * j1xsh + (x**2 - y**2) / r**3 * j1xa1
                              - x**2 / r**2 * j0x2a1)
        self.u_tensor[0][1] = (x * y / r * j1xsh + (2*x*y) / r**3 * j1xa1
                              - x * y / r**2 * j0x2a1)
        self.u_tensor[1][0] = self.u_tensor[0][1]
        self.u_tensor[1][1] = (j0sh + y**2 / r * j1xsh + (y**2 - x**2) / r**3 * j1xa1
                              - y**2 / r**2 * j0x2a1)
        
        self.u_tensor[0][2] = coeff * x / r * j1xsh + x / r * j1xa4_13
        self.u_tensor[1][2] = coeff * y / r * j1xsh + y / r * j1xa4_13
        
        self.u_tensor[2][0] = coeff * j1xsh + x / r * j1xa4_31
        self.u_tensor[2][1] = coeff * j1xsh + y / r * j1xa4_31
        
        self.u_tensor[2][2] = j0sh - j0xdsh + j0a2
        
        return self.u_tensor.copy()

    def get_Ufunc(self, point_locs):
        Ufunc = sk.np.array([self.get_ufunc(pt) for pt in point_locs])
        return (Ufunc[:, i, j] for i in range(3) for j in range(3))

    # Since the integrals in the velocity funtion are duplicate. 
    # We only need to calculate them once.
    
    def get_uxfunc(self, point_loc):
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        r = sk.np.sqrt(x**2 + y**2)
        (z_, h_) = (z, self.height) if z > self.height else (self.Height - z, self.Height - self.height)
        paraw = [r, z, self.height, self.Height]
        parav = [r, z_, h_, self.Height]
        coeff = z - self.height
        if z_ < 3. * h_:
            j0sh = self.near_integral(sk.j0sh, *parav, m=0)
            j1xsh = self.near_integral(sk.j1xsh, *parav, m=1)
        else:
            imaxv = 9. * r / (z_ - h_)
            j0sh = self.far_integral(sk.j0sh, *paraw, imaxv, method='gauss-legendre')
            j1xsh = self.far_integral(sk.j1xsh, *paraw, imaxv, method='gauss-legendre')

        imaxw = 9. * r / (z + self.height)
        j0x2a1 = self.far_integral(sk.j0x2a1, *paraw, imaxw, method='gauss-legendre')
        j1xa1 = self.far_integral(sk.j1xa1, *paraw, imaxw, method='gauss-legendre')
        j1xa4_13 = self.far_integral(sk.j1xa4_13, *paraw, imaxw, method='gauss-legendre')
        
        self.u_tensor[0][0] = (j0sh + x**2 / r * j1xsh + (x**2 - y**2) / r**3 * j1xa1
                              - x**2 / r**2 * j0x2a1)
        self.u_tensor[0][1] = (x * y / r * j1xsh + (2*x*y) / r**3 * j1xa1
                              - x * y / r**2 * j0x2a1) 
        self.u_tensor[0][2] = coeff * x / r * j1xsh + x / r * j1xa4_13
        return self.u_tensor[0, :]
       
    
    def get_uyfunc(self, point_loc):
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        r = sk.np.sqrt(x**2 + y**2)
        (z_, h_) = (z, self.height) if z > self.height else (self.Height - z, self.Height - self.height)
        paraw = [r, z, self.height, self.Height]
        parav = [r, z_, h_, self.Height]
        coeff = z - self.height
        if z_ < 3. * h_:
            j0sh = self.near_integral(sk.j0sh, *parav, m=0)
            j1xsh = self.near_integral(sk.j1xsh, *parav, m=1)
        else:
            imaxv = 9. * r / (z_ - h_)
            j0sh = self.far_integral(sk.j0sh, *paraw, imaxv, method='gauss-legendre')
            j1xsh = self.far_integral(sk.j1xsh, *paraw, imaxv, method='gauss-legendre')

        imaxw = 9. * r / (z + self.height)
        j0x2a1 = self.far_integral(sk.j0x2a1, *paraw, imaxw, method='gauss-legendre')
        j1xa1 = self.far_integral(sk.j1xa1, *paraw, imaxw, method='gauss-legendre')
        j0a2 = self.far_integral(sk.j0a2, *paraw, imaxw, method='gauss-legendre')
        j1xa4_13 = self.far_integral(sk.j1xa4_13, *paraw, imaxw, method='gauss-legendre')
        
        self.u_tensor[1][0] = (x * y / r * j1xsh + (2*x*y) / r**3 * j1xa1
                              - x * y / r**2 * j0x2a1)
        self.u_tensor[1][1] = (j0sh + y**2 / r * j1xsh + (y**2 - x**2) / r**3 * j1xa1
                              - y**2 / r**2 * j0x2a1)
        self.u_tensor[1][2] = coeff * y / r * j1xsh + y / r * j1xa4_13
        return self.u_tensor[1, :]
    
    def get_uzfunc(self, point_loc):
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        r = sk.np.sqrt(x**2 + y**2)
        (z_, h_) = (z, self.height) if z > self.height else (self.Height - z, self.Height - self.height)
        paraw = [r, z, self.height, self.Height]
        parav = [r, z_, h_, self.Height]
        coeff = z - self.height
        
        if z_ < 3. * h_:
            j0sh = self.near_integral(sk.j0sh, *parav, m=0)
            j1xsh = self.near_integral(sk.j1xsh, *parav, m=1)
            j0xdsh = self.near_integral(sk.j0xdsh, *parav, m=0)
        else:
            imaxv = 9. * r / (z_ - h_)
            j0sh = self.far_integral(sk.j0sh, *paraw, imaxv, method='gauss-legendre')
            j1xsh = self.far_integral(sk.j1xsh, *paraw, imaxv, method='gauss-legendre')
            j0xdsh = self.far_integral(sk.j0xdsh, *paraw, imaxv, method='gauss-legendre')

        imaxw = 9. * r / (z + self.height)
        j0a2 = self.far_integral(sk.j0a2, *paraw, imaxw, method='gauss-legendre')
        j1xa4_31 = self.far_integral(sk.j1xa4_31, *paraw, imaxw, method='gauss-legendre')
        
        self.u_tensor[2][0] = coeff * x / r * j1xsh + x / r * j1xa4_31
        self.u_tensor[2][1] = coeff * y / r * j1xsh + y / r * j1xa4_31
        self.u_tensor[2][2] = j0sh - j0xdsh + j0a2
        
        return self.u_tensor[2, :]
    
    # Numerica derivative. 
    def xderi(self, x, y, z, dr=1e-6):
        return self.get_uxfunc([x+dr, y, z])/(2*dr)- self.get_uxfunc([x-dr, y, z])/(2*dr)

    def yderi(self, x, y, z, dr=1e-6):
        return self.get_uyfunc([x, y+dr, z])/(2*dr)- self.get_uyfunc([x, y-dr, z])/(2*dr)

    def zderi(self, x, y, z, dr=1e-6):
        return self.get_uzfunc([x, y, z+dr])/(2*dr)- self.get_uzfunc([x, y, z-dr])/(2*dr)

    def ndiv(self, point_locs, dr=1e-6):
        xd = self.xderi(*point_locs, dr)
        yd = self.yderi(*point_locs, dr)
        zd = self.zderi(*point_locs, dr)
        return xd + yd + zd   
