import stokes_base_integral as sk
import stokes_base_series as sks 
'''
stokes_base_integral: use integral method to get the green function.
sotkes_base_series: use series method to get the green function.

The commom package is already loaded in sk. 
np -- numpy; mp -- mpmath; plt -- matplotlib; 
sb -- seaborn; pd -- pandas

scipy.special is loaded in sks.
scipy.special -- sps

'''

class tank():
    ''' we set the first plate as the plane of z=0.
	
    (x_F, y_F, z_F): the position of point force.
    height: z_F, the height of force point.
    Height: the height of the second plate.
    u_tensor: a 3*3 Green function tensor. 
	u_ij: u_i caused by F_j. 
    U_tensor: a list of u_tensor.
	For test, user can set height only then the force location is (0, 0, height).
    
	''' 
    def __init__(self, Height):
        
        self.Height = Height
        self.x_F = 0
        self.y_F = 0
        self.z_F = 0
        self.height = 0
        self.u_tensor = sk.np.zeros([3, 3]) 
        self.U_tensor = []
        '''Set local name for the global function.'''
        self.far_integral = sk.far_integral
        self.near_integral = sk.near_integral 
		
    def set_locF(self, x_F, y_F, z_F):
        '''Set the location of force.'''
        self.x_F, self.y_F, self.z_F = x_F, y_F, z_F
        self.height = self.z_F
    
    def set_height(self, height):
        self.height = height
    	
    def get_ux_ingegral(self, x, y, z, imax=4):
        '''Get Green function for ux by integral method in sk'''
        para_1 = [x, y, z, self.height, self.Height, imax]
        para_2 = [y, x, z, self.height, self.Height, imax]

        self.u_tensor[0][0] = sk.u11(*para_1)
        self.u_tensor[0][1] = sk.u12(*para_1)
        self.u_tensor[0][2] = sk.u13(*para_1)
        
        return self.u_tensor[0]
    
    def get_uy_integral(self, x, y, z, imax=4):
        '''Get Green function for uy by integral method in sk'''
        para_1 = [x, y, z, self.height, self.Height, imax]
        para_2 = [y, x, z, self.height, self.Height, imax]

        self.u_tensor[1][0] = sk.u12(*para_1)
        self.u_tensor[1][1] = sk.u11(*para_2)
        self.u_tensor[1][2] = sk.u13(*para_2)
        
        return self.u_tensor[1]
    
    def get_uz_integral(self, x, y, z, imax=4):
        '''Get Green function for uz by integral method in sk'''
        para_1 = [x, y, z, self.height, self.Height, imax]
        para_2 = [y, x, z, self.height, self.Height, imax]

        self.u_tensor[2][0] = sk.u31(*para_1)
        self.u_tensor[2][1] = sk.u31(*para_2) 
        self.u_tensor[2][2] = sk.u33(*para_1)
        
        return self.u_tensor[2]
    
    def get_u_integral(self, x, y, z, imax=4):
        '''Get Green function for one point by integral method in sk'''
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
        '''Get the full space Green function by integral in sk.'''
        if (type(x_list) or type(y_list) or type(z_list)) is not sk.np.ndarray:
            raise TypeError("x, y, z should be array!")
        
        Uxy = sk.np.array([[[self.get_u(xx, yy, zz, imax) 
                          for xx in x_list] 
                          for yy in y_list]
                          for zz in z_list])
        return Uxy
	
    '''Added in 8.7, 2017. Reorganize the integral part then it is efficient 
    in performance but tedious in code.'''
    def get_uxfunc_integral(self, point_loc):
        '''Return the series solution of Green funcition caused by Fx.
        
        Arguments:
        point_loc -> location of field point
        n_f: final number of your series.
        n_i: initial number of your series. 
        
        '''
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
       
    
    def get_uyfunc_integral(self, point_loc):
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
    
    def get_uzfunc_integral(self, point_loc):
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

    def get_ufunc_integral_far(self, point_loc): 
        '''      
        First We get the coordinate of point relative to force,
        e.g, (x-x_F, y-y_F, z). (Not z-z_F!)
        When z > h, we use oscillation integral(near_integral) to 
        get the green funciton if z < 3h and normal integral(far-
        integral) else.
        For the z < h case, just replace the z and h to H-z and H-z
        under the integral.
        '''
        
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
        
        self.u_tensor[2][0] = coeff * x / r * j1xsh + x / r * j1xa4_31
        self.u_tensor[2][1] = coeff * y / r * j1xsh + y / r * j1xa4_31
        
        self.u_tensor[2][2] = j0sh - j0xdsh + j0a2
        
        return self.u_tensor.copy()

    def get_ufunc_integral_near(self, point_loc): 
        '''      
        First We get the coordinate of point relative to force,
        e.g, (x-x_F, y-y_F, z). (Not z-z_F!)
        When z > h, we use oscillation integral(near_integral) to 
        get the green funciton if z < 3h and normal integral(far-
        integral) else.
        For the z < h case, just replace the z and h to H-z and H-z
        under the integral.
        '''
        
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
        j0x2a1 = sk.near_integral(sk.j0x2a1, *paraw, m=0)
        j1xa1 = near_integral(sk.j1xa1, *paraw, m=1)
        j0a2 = near_integral(sk.j0a2, *paraw, m=0)
        j1xa4_13 = near_integral(sk.j1xa4_13, *paraw, m=1)
        j1xa4_31 = near_integral(sk.j1xa4_31, *paraw, m=1)
        
        self.u_tensor[0][0] = (j0sh + x**2 / r * j1xsh + (x**2 - y**2) / r**3 * j1xa1
                              - x**2 / r**2 * j0x2a1)
        self.u_tensor[0][1] = (x * y / r * j1xsh + (2*x*y) / r**3 * j1xa1
                              - x * y / r**2 * j0x2a1)
        self.u_tensor[1][0] = self.u_tensor[0][1]
        self.u_tensor[1][1] = (j0sh + y**2 / r * j1xsh + (y**2 - x**2) / r**3 * j1xa1
                              - y**2 / r**2 * j0x2a1)
        
        self.u_tensor[0][2] = coeff * x / r * j1xsh + x / r * j1xa4_13
        self.u_tensor[1][2] = coeff * y / r * j1xsh + y / r * j1xa4_13
        
        self.u_tensor[2][0] = coeff * x / r * j1xsh + x / r * j1xa4_31
        self.u_tensor[2][1] = coeff * y / r * j1xsh + y / r * j1xa4_31
        
        self.u_tensor[2][2] = j0sh - j0xdsh + j0a2
        return self.u_tensor.copy()

    def get_Ufunc_integral_far(self, point_locs):
        '''Return a generator that generates 9 component of Green function.'''
        Ufunc = sk.np.array([self.get_ufunc_integral_far(pt) for pt in point_locs])
        return (Ufunc[:, i, j] for i in range(3) for j in range(3))

    def get_Ufunc_integral_near(self, point_locs):
        '''Return a generator that generates 9 component of Green function.'''
        Ufunc = sk.np.array([self.get_ufunc_integral_near(pt) for pt in point_locs])
        return (Ufunc[:, i, j] for i in range(3) for j in range(3))

    def xderi(self, x, y, z, dr=1e-6):
        '''Return numerica derivative of uxfunc.'''
        return self.get_uxfunc([x+dr, y, z])/(2*dr)- self.get_uxfunc([x-dr, y, z])/(2*dr)

    def yderi(self, x, y, z, dr=1e-6):
        '''Return numerica derivative of uyfunc.'''
        return self.get_uyfunc([x, y+dr, z])/(2*dr)- self.get_uyfunc([x, y-dr, z])/(2*dr)

    def zderi(self, x, y, z, dr=1e-6):
        '''Return numerica derivative of uzfunc'''
        return self.get_uzfunc([x, y, z+dr])/(2*dr)- self.get_uzfunc([x, y, z-dr])/(2*dr)

    def ndiv(self, point_locs, dr=1e-6):
        xd = self.xderi(*point_locs, dr)
        yd = self.yderi(*point_locs, dr)
        zd = self.zderi(*point_locs, dr)
        return xd + yd + zd   

    '''Added in 8.17, 2017. Input array, return full Green funciton
    of the series type.'''
    def get_ux_series(self, point_loc, n_f=50, n_i=1):
        '''Return the series solution of Green funcition caused by Fx.
        
        Arguments:
        point_loc -> location of field point
        n_f: final number of your series.
        n_i: initial number of your series. 
        
        '''
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        #r = sk.np.sqrt(x**2 + y**2)
        para1 = [x, y, z, self.height, Height, n_i, n_f]
        self.u_tensor[0][0] = sks.u11(*para1)
        self.u_tensor[0][1] = sks.u12(*para1)
        self.u_tensor[0][2] = sks.u13(*para1)
        return self.u_tensor[0, :]
    
    def get_uy_series(self, point_loc, n_f=50, n_i=1):
        '''Return the series solution of Green funcition caused by Fx.
        
        Arguments:
        point_loc -> location of field point
        n_f: final number of your series.
        n_i: initial number of your series. 
        
        '''
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        #r = sk.np.sqrt(x**2 + y**2)
        '''u21(x, y) = u12(y, x)'''
        para2 = [y, x, z, self.height, Height, n_i, n_f]
        self.u_tensor[1][0] = sks.u12(*para2)
        self.u_tensor[1][1] = sks.u11(*para2)
        self.u_tensor[1][2] = sks.u13(*para2)
        return self.u_tensor[1, :]
        
    def get_uz_series(self, point_loc, n_f=50, n_i=1):
        '''Return the series solution of Green funcition caused by Fx.
        
        Arguments:
        point_loc -> location of field point
        n_f: final number of your series.
        n_i: initial number of your series. 
        
        '''
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        #r = sk.np.sqrt(x**2 + y**2)
        '''u32(x, y) = u31(y, x)'''
        para1 = [x, y, z, self.height, Height, n_i, n_f]
        para2 = [y, x, z, self.height, Height, n_i, n_f]
        self.u_tensor[2][0] = sks.u31(*para1)
        self.u_tensor[2][1] = sks.u31(*para2)
        self.u_tensor[2][2] = sks.u33(*para1)
        return self.u_tensor[2, :]
    
    def get_u_series(self, point_loc, n_f=50, n_i=1):
        self.get_ux_series(point_loc, n_f, n_i)
        self.get_uy_series(point_loc, n_f, n_i)
        self.get_uz_series(point_loc, n_f, n_i)
        return self.u_tensor.copy()

    def get_uxfunc_series(self, point_loc, n_f=50, n_s=1):
        '''Return the series solution of Green funcition caused by Fx.
        
        Arguments:
        point_loc -> location of field point, array
        n_f: final number of your series.
        n_s: start number of your series. 
        
        '''
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        #r = sk.np.sqrt(x**2 + y**2)
        para1 = [x, y, z, self.height, self.Height, n_f, n_s]
        self.u_tensor[0][0] = sks.u11(*para1)
        self.u_tensor[0][1] = sks.u12(*para1)
        self.u_tensor[0][2] = sks.u13(*para1)
        return self.u_tensor[0, :]
    
    def get_uyfunc_series(self, point_loc, n_f=50, n_s=1):
        '''Return the series solution of Green funcition caused by Fx.
        
        Arguments:
        point_loc -> location of field point, array
        n_f: final number of your series.
        n_s: initial number of your series. 
        
        '''
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        #r = sk.np.sqrt(x**2 + y**2)
        '''u21(x, y) = u12(y, x)'''
        para2 = [y, x, z, self.height, self.Height, n_f, n_s]
        self.u_tensor[1][0] = sks.u12(*para2)
        self.u_tensor[1][1] = sks.u11(*para2)
        self.u_tensor[1][2] = sks.u13(*para2)
        return self.u_tensor[1, :]
        
    def get_uzfunc_series(self, point_loc, n_f=50, n_s=1):
        '''Return the series solution of Green funcition caused by Fx.
        
        Arguments:
        point_loc -> location of field point, array
        n_f: final number of your series.
        n_s: initial number of your series. 
        
        '''
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        #r = sk.np.sqrt(x**2 + y**2)
        '''u32(x, y) = u31(y, x)'''
        para1 = [x, y, z, self.height, self.Height, n_f, n_s]
        para2 = [y, x, z, self.height, self.Height, n_f, n_s]
        self.u_tensor[2][0] = sks.u31(*para1)
        self.u_tensor[2][1] = sks.u31(*para2)
        self.u_tensor[2][2] = sks.u33(*para1)
        return self.u_tensor[2, :]
    
    def get_ufunc_series(self, point_loc, n_f=50, n_s=1):
        '''Return the series solution of  full Green funcition.
        
        Arguments:
        point_loc -> location of field point, array
        n_f: final number of your series.
        n_s: initial number of your series. 

        '''
        x, y, z= point_loc[0]-self.x_F, point_loc[1]-self.y_F, point_loc[2]
        x3 = z
        h = self.height
        H = self.Height
        pi = sks.np.pi
        r = sk.np.sqrt(x**2 + y**2)
        zm = sks.zm
        sinsin = sks.np.array([sks.sinsin(x3, h, H, n) for n in range(n_s, n_f+1)])
        shsh = sks.np.array([sks.shsh(x3, h, H, n) for n in range(n_s, n_f+1)])
        chsh = sks.np.array([sks.chsh(x3, h, H, n) for n in range(n_s, n_f+1)])
        shch = sks.np.array([sks.shch(x3, h, H, n) for n in range(n_s, n_f+1)])
        shMinch = sks.np.array([sks.shMinch(x3, h, H, n) for n in range(n_s, n_f+1)])
        chMinsh = sks.np.array([sks.chMinsh(x3, h, H, n) for n in range(n_s, n_f+1)])
        zm_Add = sks.np.array([sks.zm_Add(n) for n in range(n_s, n_f+1)])
        inverseOfzm_Min = sks.np.array([sks.inverseOfzm_Min(n) for n in range(n_s, n_f+1)])
        hankel0 = sks.np.array([sks.mp.hankel1(0, r*zm(n)/H) for n in range(n_s, n_f+1)])
        hankel1 = sks.np.array([sks.mp.hankel1(1, r*zm(n)/H) for n in range(n_s, n_f+1)])
        besselk0 = sks.np.array([sks.sps.k0(r*n*pi/H) for n in range(n_s, n_f+1)])
        besselk1= sks.np.array([1 / (n*pi) * sks.sps.k1(r*n*pi/H) for n in range(n_s, n_f+1)])
    
        uabH0 = sks.np.array([sks.mp.im(pi * zm(n+1)/H * hankel0[n] * (1/zm(n+1)*shsh[n] + chsh[n] + shch[n]
                    - inverseOfzm_Min[n]*zm(n+1)*((x3+h)/H*shsh[n]
                    + x3*h/H**2*(sks.mp.cosh((h-x3)*zm(n+1)/H)-chMinsh[n])))) 
                    for n in range(n_f)])
        uabH1 = sks.np.array([sks.mp.im(pi * hankel1[n] * 
                    (1/zm(n+1)*shsh[n] + chsh[n] + shch[n]
                     - inverseOfzm_Min[n]*zm(n+1)*((x3+h)/H*shsh[n]
                    + x3*h/H**2*(sks.mp.cosh((h-x3)*zm(n+1)/H)-chMinsh[n]))))
                    for n in range(n_f)])
        uabK0 = 4./ H * sinsin * besselk0
        uabK1 = 4. * sinsin * besselk1
        ua3H1 = sks.np.array([-pi/H * sks.mp.im(zm(n+1) * hankel1[n] * inverseOfzm_Min[n] 
                    * (x3*h*zm(n+1)/H**2 *(sks.mp.sinh((x3-h)*zm(n+1)/H) + shMinch[n]) 
                    + zm(n+1)*(chsh[n] - shch[n])
                    + shsh[n]*((h-x3)/H*(zm_Add[n]-1) - ((x3+h)/H - 1))))
                    for n in range(n_f)])
        u3aH1 = sks.np.array([-pi/H * sks.mp.im(zm(n+1) * hankel1[n] * inverseOfzm_Min[n] 
                    * (x3*h*zm(n+1)/H**2 *(sks.mp.sinh((x3-h)*zm(n+1)/H) - shMinch[n]) 
                    + zm(n+1)*(chsh[n] - shch[n])
                    + shsh[n]*((h-x3)/H*(zm_Add[n]-1) + ((x3+h)/H - 1))))
                    for n in range(n_f)])
        u33H0 = sks.np.array([-pi/H * sks.mp.im(zm(n+1) * hankel0[n] * inverseOfzm_Min[n]
                    * (zm_Add[n]*(chsh[n] + shch[n] - 1/zm(n+1) * shsh[n])
                    - (h*x3)/(H**2)*zm(n+1)*(chMinsh[n] + sks.mp.cosh((x3 - h)*zm(n+1)/H))
                    - zm(n+1)*(x3 + h)/H*shsh[n]))
                    for n in range(n_f)])
        
        
        self.u_tensor[0][0] = ((x**2/r**2 * (uabH0 + uabK0) 
                                + (y**2-x**2)/r**3 * (uabH1+ uabK1 + r*uabK0)).sum()
                                - (y**2-x**2)/r**4 * sks.uabSimple(r, x3, h, H))
        self.u_tensor[0][1] = ((x*y/r**2 * (uabH0 + uabK0)
                                + (-2*x*y)/r**3 * (uabH1 + uabK1+ r*uabK0)).sum()
                                + (2*x*y)/r**4 * sks.uabSimple(r, x3, h, H))
        self.u_tensor[1][0] = self.u_tensor[0][1]
        self.u_tensor[1][1] = ((y**2/r**2 * (uabH0 + uabK0) 
                                + (x**2-y**2)/r**3 * (uabH1+ uabK1 + r*uabK0)).sum()
                                - (x**2-y**2)/r**4 * sks.uabSimple(r, x3, h, H))
        
        self.u_tensor[0][2] = x / r * ua3H1.sum()
        self.u_tensor[1][2] = y / r * ua3H1.sum()
        
        self.u_tensor[2][0] = x / r * u3aH1.sum()
        self.u_tensor[2][1] = y / r * u3aH1.sum()
        
        self.u_tensor[2][2] = u33H0.sum()
        return self.u_tensor.copy()
    
    def get_Ufunc_series(self, point_locs):
        '''Return a generator that generates 9 component of Green function.'''
        Ufunc = sk.np.array([self.get_ufunc_series(pt) for pt in point_locs])
        return (Ufunc[:, i, j] for i in range(3) for j in range(3))