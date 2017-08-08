from sympy import symbols, diff, lambdify
from sympy import sinh,cosh, besselj
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import mpmath as mp
import pandas as pd

mp.dps = 15
mp.pretty = True

'''

It's a file that contains the funcions used in stokes_main.py.
Here the index of v and w obeys the following rule.
The  Green function compnent vij = ith component of v caused by jth component of force.
So the ith velocity component is V_i = v_ij * F_j
Since the x-y plane is infinitely large, the green funciton is symmetric of x, y which means
u21(x, y) = u12(y, x). For this consideration, we only define five componen of Green fucntion tensor.

The relation of u, v and w.
Real velocity field is u. v and w represent two different contributions.
So u = v + w.
v is rooted from image method while w is the correction term in order to satisfity boundary condition.

'''


# Step1: Define funcitons and variables.

# H is the height of the second disk. h is the height of the location of force.
# x3 is the z-component of a point in the fluid.
# r^2 = x1^2 + x2^2. 
# In a single calucation, H = const, h=const, we want to find the pressure and velocity field. 


''' 
The use of sympy is for the derivatives.
After we define the symbolic expression of a1..., we lambdify them.
'''
x, h, H, x3, r = symbols('x, h, H, x3, r')  
rational_sinh = sinh(h*x)/sinh(H*x)
d_rational_sinh = diff(rational_sinh,x)
sh = sinh(h*x) / sinh(H*x) * sinh((H-x3)*x)
d_sh = diff(sh, x)
zeros_j = np.load('zeros_j.npy')

a1 = 1/(sinh(x*H)**2 - (x*H)**2)*(
    x*h*H*x3*sinh(x*(H - x3))*sinh(x*(H - h))
     + x3*(h*sinh(x*H)*cosh(x*(H - x3 - h)) - H*sinh(x*h)*cosh(x3*x))
     + x3*x*H*sinh(x*H)*cosh(x*(H - x3))*d_rational_sinh.subs([(h,H-h)])
     - H*sinh(x*x3)*(sinh(x*H)*d_rational_sinh +  x*H*d_rational_sinh.subs([(h,H-h)])))

a2 = x**2 / (sinh(x*H)**2 - (x*H)**2) * (
        x3*(H*sinh(x*h)*cosh(x*x3) - h*sinh(x*H)*cosh(x*(H-x3-h))
        + x*h*H*sinh(x*(H-x3))*sinh(x*(H-h))
        + x*H*cosh(x*(H-x3))*sinh(x*H)*d_rational_sinh.subs([(h, H-h)]))
        - x*H**2*sinh(x*x3)*d_rational_sinh.subs([(h, H-h)]) 
        + H*sinh(x*x3)*sinh(x*H)*d_rational_sinh)


a3 = (h*sinh(x*H)*cosh(x*(H-x3-h)) - H*sinh(x*h)*cosh(x*(H+x3))) / (sinh(x*H)**2)

a4_13= x/(sinh(x*H)**2 - (x*H)**2)*(x3*x*H*(h*sinh(x*(x3 - h))
        + H*sinh(x*h)*sinh(x*(H - x3))/sinh(x*H))
        - x*h*H**2*sinh(x*x3)*sinh(x*(H - h))/sinh(x*H)
        + (-1)*(-H*(H - h)*sinh(x*h)*sinh(x*x3) 
        + x3*(h*sinh(x*H)*sinh(x*(H - x3 - h))
        + H*sinh(x*h)*sinh(x*x3))))
           
a4_31= x/(sinh(x*H)**2 - (x*H)**2)*(x3*x*H*(h*sinh(x*(x3 - h))
        + H*sinh(x*h)*sinh(x*(H - x3))/sinh(x*H))
        - x*h*H**2*sinh(x*x3)*sinh(x*(H - h))/sinh(x*H)
        + (-H*(H - h)*sinh(x*h)*sinh(x*x3) + x3*(h*sinh(x*H)*sinh(x*(H - x3 - h))
        + H*sinh(x*h)*sinh(x*x3))))

a5 = x**2 / (sinh(x*H)**2-(x*H)**2) * (
    (sinh(x*H)*d_rational_sinh + x*H*d_rational_sinh.subs([(h, H-h)])) * sinh(x*H)*sinh(x*(H-x3)) 
    + (x*h*H*sinh(x*(H-h)) + (H-h)*sinh(x*h)*sinh(x*H)) * cosh(x*(H-x3)))

a6 = x**2 / (sinh(x*H)**2-(x*H)**2) * (
    (x*h*H*sinh(x*(H - h)) - (H - h)*sinh(x*h)*sinh(x*H))*sinh(x*(H - x3)) 
    -(sinh(x*H)*d_rational_sinh - x*H*d_rational_sinh)*sinh(x*H)*cosh(x*(H - x3)))


# we lambdify them to mpmath functions. 
a1_ = lambdify([x, x3, h, H],a1,('mpmath', 'numpy'))
a2_ = lambdify([x, x3, h, H],a2, ('mpmath', 'numpy'))
a3_ = lambdify([x, x3, h, H],a3, ('mpmath', 'numpy'))
a4_31 = lambdify([x, x3, h, H],a4_31, ('mpmath', 'numpy'))
a4_13 = lambdify([x, x3, h, H], a4_13, ('mpmath', 'numpy'))
a5_ = lambdify([x, x3, h, H],a5, ('mpmath', 'numpy'))
a6_ = lambdify([x, x3, h, H],a6, ('mpmath', 'numpy'))

sh_ = lambdify([x, x3, h, H], sh, ('mpmath', 'numpy'))
d_sh_ = lambdify([x, x3, h, H], d_sh, ('mpmath', 'numpy'))


# xz is the hight of field point.
# radius**2 = x**2 + y**2
# hight is the hight of point force.
# Hight is the hight of second disk.

# h0 is the lenght scale.
# k is the recipocal vector. So the integral parameter is dimensionless.

h0 = 1.
k0 = 2. * mp.pi / h0

def j0sh(x, radius, xz, height, Height):
    k = k0 / radius
    return k * mp.j0(2.*np.pi*x) * sh_(x*k, xz*h0, height*h0, Height*h0)


def j1xsh(x, radius, xz, height, Height):
    k = k0 / radius
    return k * mp.j1(2.*np.pi*x) * (k*x) * sh_(x*k, xz*h0, height*h0, Height*h0)

def j0xdsh(x, radius, xz, height, Height):
    k = k0 / radius
    return k * mp.j0(2.*np.pi*x) * (k*x) * d_sh_(x*k, xz*h0, height*h0, Height*h0)
	
def j1xa1(x, radius, xz, height, Height):
    k = k0 / radius
    return k * mp.j1(2.*np.pi*x) * (k*x) * a1_(x*k, xz*h0, height*h0, Height*h0)

def j0x2a1(x, radius, xz, height, Height):
    k = k0 / radius
    return k * mp.j0(2.*np.pi*x) * (k*x)**2 * a1_(x*k, xz*h0, height*h0, Height*h0)

def j0a2(x, radius, xz, height, Height):
    k = k0 / radius
    return k * mp.j0(2.*np.pi*x) * a2_(k*x, xz*h0, height*h0, Height*h0) 

def j1xa4_13(x, radius, xz, height, Height):
    k = k0 / radius
    return k * mp.j1(2.*np.pi*x) * (k*x) * a4_13(k*x, xz*h0, height*h0, Height*h0) 

def j1xa4_31(x, radius, xz, height, Height):
    k = k0 / radius
    return k * mp.j1(2.*np.pi*x) * (k*x) * a4_31(k*x, xz*h0, height*h0, Height*h0) 

# Functions with no 'a' are hard to integral. We seperate them out.
func_list_v = [j0sh, j1xsh, j0xdsh]
func_list_u = [j0x2a1, j1xa1, j0a2, j1xa4_13, j1xa4_31]

# Step2: Define the velocity function.
# Since every velocity contains an integral, and the integral is hard to evaluate around z=h
# which is the xy plane that point force locates, we define two funcitions, 
# one for far-distance integral, and the other for near-h integral.

# Note: We give the explicit number of arguments for we only need these parameters.
#       For general case, change the arguments to *args, and add *kwrds.

# far_integral use sinh-tanh method since the integral envolves sinh.
# However, gauss-legendre method maybe more efficient with equal precision.

# near_integral use mp.quadosc which is designed for oscillation integrals in infinity interval.
# zeros of the integrated function must be given in the function form. 
# Here I chose the builtin function: besseljzero.
# Another option is making a list of zeros then make reference to them. 
# It's efficient but may not be accurate when the desired number of zeors are huge.

def far_integral(func, radius, z, hight, Height, imax, **kwards):
    return np.array(mp.quad(lambda x: func(x, radius, z, hight, Height), 
					np.linspace(0, imax, 3), **kwards))

def near_integral(func, radius, z, hight, Height, m=0):
    return np.array(mp.quadosc(lambda x: func(x, radius, z, hight, Height), 
                    [0, mp.inf], zeros=lambda n: zeros_j[m][int(n-1)] / (2*mp.pi)))



# Following are velocity functions.

# imax: the integral upper bound.
# The default value of imax is 4.0 which is proved to be accurate enough for far_integral. 

# Velocity funcitons
# Every funciton has the claim r = np.sqrt(x**2 + y**2). It must need simplify such as decorator.

# u11 ========================================= 

def v11(x, y, z, h, H, imax=4.0):
    (z, h) = (z, h) if z > h else (H - z, H - h)
    
    #r = np.sqrt(x**2 + y**2)
    if z > 3 * h:
        imax = 9. * r / (z - h)
        return  (far_integral(j0sh, r, z, h, H, imax, method='gauss-legendre') + 
                x**2 / r * h0 * far_integral(j1xsh, r, z, h, H, imax, method='gauss-legendre'))
    else:
        return  (near_integral(j0sh, r, z, h, H, m=0) + 
                x**2 / r * h0 * near_integral(j1xsh, r, z, h, H, m=1))

def w11(x, y, z, h, H, imax=4.0):
    #r = np.sqrt(x**2 + y**2)
    imax = 9. * r / (z + h)
    return ((x**2 - y**2) / (r**3 * np.sqrt(h0)) * far_integral(j1xa1, r, z, h, H, imax, method='gauss-legendre') - 
            x**2 / r**2 * far_integral(j0x2a1, r, z, h, H, imax, method='gauss-legendre'))

# u12 =========================================

def v12(x, y, z, h, H, imax=4.0):
    (z, h) = (z, h) if z > h else (H - z, H - h)
    #r = np.sqrt(x**2 + y**2)
    if z > 3 * h:
        imax = 9. * r / (z - h)
        return x * y / r * h0 * far_integral(j1xsh, r, z, h, H, imax, method='gauss-legendre')
    else:
        return x * y / r * h0 * near_integral(j1xsh, r, z, h, H, m=1)

def w12(x, y, z, h, H, imax=4.0):
    #r = np.sqrt(x**2 + y**2)
    imax = 9. * r / (z + h)
    return ((2*x*y) / (r**3 * np.sqrt(h0)) * far_integral(j1xa1, r, z, h, H, imax, method='gauss-legendre') - 
            x * y / r**2 * far_integral(j0x2a1, r, z, h, H, imax, method='gauss-legendre'))

# u13 =========================================

# =====================================================
# The piecewise function of v13 needs further consideration.
# z = h may never hold for the float presicion.
def v13(x, y, z, h, H, imax=4.0):
    # The coefficient of integrals. 
    coeff = (z - h) * x / r * h0
    (z, h) = (z, h) if z > h else (H - z, H - h)
    #r = np.sqrt(x**2 + y**2)
    if z > 3 * h:
        imax = 9. * r / (z - h)
        return coeff * far_integral(j1xsh, r, z, h, H, imax, method='gauss-legendre')
    elif z == h :
        return 0
    else:
         return coeff * near_integral(j1xsh, r, z, h, H, m=1)

def w13(x, y, z, h, H, imax=8.0):
    #r = np.sqrt(x**2 + y**2)
    imax = 9. * r / (z + h)
    return x / r * far_integral(j1xa4_13, r, z, h, H, imax, method='gauss-legendre')

# u31 =========================================

def v31(x, y, z, h, H, imax=4.0):
    return v13(x, y, z, h, H, imax=4.0)

def w31(x, y, z, h, H, imax=4.0):
    #r = np.sqrt(x**2 + y**2)
    imax = 9. * r / (z + h)
    return x / r * far_integral(j1xa4_31, r, z, h, H, imax, method='gauss-legendre')

# u33 =========================================

def v33(x, y, z, h, H, imax=4.0):
    (z, h) = (z, h) if z > h else (H - z, H - h)
    #r = np.sqrt(x**2 + y**2)
    if z > 3 * h : 
        imax = 9. * r / (z - h)
        return (far_integral(j0sh, r, z, h, H, imax, method='gauss-legendre') 
               - far_integral(j0xdsh, r, z, h, H, imax, method='gauss-legendre'))
    else:
        return (near_integral(j0sh, r, z, h, H, m=0) 
                - near_integral(j0xdsh, r, z, h, H, m=0))
    
    
def w33(x, y, z, h, H, imax=4.0):
    imax = 9. * r / (z + h)
    #r = np.sqrt(x**2 + y**2)
    return far_integral(j0a2, r, z, h, H, imax, method='gauss-legendre')


# u11 =========================================

def u11(x, y, z, h, H, imax=4.0):
    global r 
    r = np.sqrt(x**2 + y**2)
    return w11(x, y, z, h, H, imax) + v11(x, y, z, h, H, imax)                     

# u12 =========================================

def u12(x, y, z, h, H, imax=4.0):
    global r    
    r = np.sqrt(x**2 + y**2)
    return w12(x, y, z, h, H, imax) + v12(x, y, z, h, H, imax)                     

# u13 =========================================

def u13(x, y, z, h, H, imax=4.0):
    global r    
    r = np.sqrt(x**2 + y**2)
    return w13(x, y, z, h, H, imax) + v13(x, y, z, h, H, imax)                     

# u31 =========================================

def u31(x, y, z, h, H, imax=4.0):
    global r    
    r = np.sqrt(x**2 + y**2)
    return w31(x, y, z, h, H, imax) + v31(x, y, z, h, H, imax)                     
                    
# u33 =========================================

def u33(x, y, z, h, H, imax=4.0):
    global r    
    r = np.sqrt(x**2 + y**2)
    return w33(x, y, z, h, H, imax) + v33(x, y, z, h, H, imax) 
                    

'''Decorate the velocity function as vector function。'''
# U11 =========================================

def U11_z(x, y, z_list, h, H, imax=4.0):
    if (type(x) or type(y)) is not float:
        raise TypeError("x and y should be float!")
    elif type(z_list) is not np.ndarray:
        raise TypeError("z should be array!")
    global r    
    r = np.sqrt(x**2 + y**2)
    return np.array([w11(x, y, zz, h, H, imax) + v11(x, y, zz, h, H, imax) 
                    for zz in z_list])

# U12 =========================================

def U12_z(x, y, z_list, h, H, imax=4.0):
    if (type(x) or type(y)) is not float:
        raise TypeError("x and y should be float!")
    elif type(z_list) is not np.ndarray:
        raise TypeError("z should be array!")
    global r    
    r = np.sqrt(x**2 + y**2)
    return np.array([w12(x, y, zz, h, H, imax) + v12(x, y, zz, h, H, imax) 
                    for zz in z_list])

# U13 =========================================

def U13_z(x, y, z_list, h, H, imax=4.0):
    if (type(x) or type(y)) is not float:
        raise TypeError("x and y should be float!")
    elif type(z_list) is not np.ndarray:
        raise TypeError("z should be array!")
    global r    
    r = np.sqrt(x**2 + y**2)
    return np.array([w13(x, y, zz, h, H, imax) + v13(x, y, zz, h, H, imax) 
                    for zz in z_list])

# U31 =========================================

def U31_z(x, y, z_list, h, H, imax=4.0):
    if (type(x) or type(y)) is not float:
        raise TypeError("x and y should be float!")
    elif type(z_list) is not np.ndarray:
        raise TypeError("z should be array!")
    global r    
    r = np.sqrt(x**2 + y**2)
    return np.array([w31(x, y, zz, h, H, imax) + v31(x, y, zz, h, H, imax) 
                    for zz in z_list])
                    
# U33 =========================================

def U33_z(x, y, z_list, h, H, imax=4.0):
    if (type(x) or type(y)) is not float:
        raise TypeError("x and y should be float!")
    elif type(z_list) is not np.ndarray:
        raise TypeError("z should be array!")
    global r    
    r = np.sqrt(x**2 + y**2)
    return np.array([w33(x, y, zz, h, H, imax) + v33(x, y, zz, h, H, imax) 
                    for zz in z_list])
                    
