# In fact, hankel funciton can be imported by scipy.
# That maybe faster.
# z(m) is the m-th root of sinh(z)**2 = z**2 
# pre-calculate pi first.
import mpmath as mp
import numpy as np
from mpmath import sinh, cosh, sqrt
import scipy.special as sps
#import time as time


mp.dps = 30
mp.pretty = True

pi = mp.pi
zm_list = np.load('zm_list.npy')
zm = lambda n: zm_list[int(n-1)]

def sinsin(x3, h, H, m):
    return mp.sin(m*pi*h/H) * mp.sin(m*pi*x3/H)
        
def shsh(x3, h, H, m):
    return sinh(zm(m)*x3/H) * sinh(zm(m)*h/H)

def chsh(x3, h, H, m):
    return x3/H * cosh(zm(m)*x3/H)*sinh(zm(m)*h/H)
    
def shch(x3, h, H, m):
    return h/H * sinh((zm(m)*x3)/H)*cosh((zm(m)*h)/H)
    
def chMinsh(x3, h, H, m): 
    return sqrt(1+zm(m)**2) * cosh((zm(m)*(x3 + h))/H) - zm(m)*sinh((zm(m)*(h + x3))/H)

def shMinch(x3, h, H, m):
    return sqrt(1+zm(m)**2) * sinh((zm(m)*(x3 + h))/H) - zm(m)*cosh((zm(m)*(h + x3))/H)
    
def inverseOfzm_Min(n):
    return 1 / (sqrt(1+zm(n)**2)-1)

def zm_Add(n):
    return (sqrt(1+zm(n)**2)+1)

def uabHankel0(r, x3, h, H, n):
    return mp.im(pi * zm(n)/H * mp.hankel1(0, r*zm(n)/H) * (1/zm(n)*shsh(x3, h, H, n) + chsh(x3, h, H, n) + shch(x3, h, H, n)
                                            - inverseOfzm_Min(n)*zm(n)*((x3+h)/H*shsh(x3, h, H, n) 
                                            + x3*h/H**2*(cosh((h-x3)*zm(n)/H)-chMinsh(x3, h, H, n)))))

def uabHankel1(r, x3, h, H, n):
    return mp.im(pi * mp.hankel1(1, r*zm(n)/H) * (1/zm(n)*shsh(x3, h, H, n) + chsh(x3, h, H, n) + shch(x3, h, H, n)
                                            - inverseOfzm_Min(n)*zm(n)*((x3+h)/H*shsh(x3, h, H, n) 
                                            + x3*h/H**2*(cosh((h-x3)*zm(n)/H)-chMinsh(x3, h, H, n)))))
def ua3Hankel(r, x3, h, H, n):
    return -pi/H * mp.im(zm(n) * mp.hankel1(1, r*zm(n)/H) * inverseOfzm_Min(n) * (x3*h*zm(n)/H**2
                *(sinh((x3-h)*zm(n)/H) + shMinch(x3, h, H, n)) 
                + zm(n)*(chsh(x3, h, H, n) - shch(x3, h, H, n))
                + shsh(x3, h, H, n)*((h-x3)/H*sqrt(1+zm(n)**2) - ((x3+h)/H - 1))))

def u3aHankel(r, x3, h, H, n):
    return -pi/H * mp.im(zm(n) * mp.hankel1(1, r*zm(n)/H) * inverseOfzm_Min(n) * (x3*h*zm(n)/H**2
                *(sinh((x3-h)*zm(n)/H) - shMinch(x3, h, H, n)) 
                  + zm(n)*(chsh(x3, h, H, n) - shch(x3, h, H, n))
                  + shsh(x3, h, H, n)*((h-x3)/H*sqrt(1+zm(n)**2) + ((x3+h)/H - 1))))

def u33Hankel(r, x3, h, H, n):
    return -pi/H * mp.im((zm(n) * mp.hankel1(0, (r*zm(n))/H))*inverseOfzm_Min(n) *
             (zm_Add(n)*(chsh(x3, h, H, n) + shch(x3, h, H, n) - 1/zm(n) * shsh(x3, h, H, n))
            - (h*x3)/(H**2)*zm(n)*(chMinsh(x3, h, H, n) + cosh((x3 - h)*zm(n)/H))
            - zm(n)*(x3 + h)/H*shsh(x3, h, H, n)))

def uabK1(r, x3, h, H, n):
    return 4./(n*pi) * sinsin(x3, h, H, n) * mp.besselk(1, r*n*pi/H)

def uabK0(r, x3, h, H, n):
    return 4./ H * sinsin(x3, h, H, n) * mp.besselk(0, r*n*pi/H)

def uabSimple(r, x3, h, H):
    return 6. * x3 * h/H * (1. - x3/H) * (1 - h/H)

def u11_sum(r1, r2, x3, h, H, n_f, n_s=1):
    r = mp.sqrt(r1**2 + r2**2)
    return mp.nsum(lambda n: (r1**2/r**2 * (uabHankel0(r, x3, h, H, n) + uabK0(r, x3, h, H, n)) 
                    + (r2**2-r1**2)/r**3 * (uabHankel1(r, x3, h, H, n) + uabK1(r, x3, h, H, n) 
                    + r*uabK0(r, x3, h, H, n))), [int(n_s), int(n_f)])
    
'''============ Green function==========='''

def u33(r1, r2, x3, h, H, n_f, n_s=1): 
    r = mp.sqrt(r1**2 + r2**2)
    return mp.nsum(lambda n: u33Hankel(r, x3, h, H, n), [int(n_s), int(n_f)])

def u11(r1, r2, x3, h, H, n_f, n_s=1):
    r = mp.sqrt(r1**2 + r2**2)
    return u11_sum(r1, r2, x3, h, H, n_f, n_s) - (r2**2-r1**2)/r**4 * uabSimple(r, x3, h, H)

def u12(r1, r2, x3, h, H, n_f, n_s=1):
    r = mp.sqrt(r1**2 + r2**2)
    return (mp.nsum(lambda n: (r1*r2/r**2 * (uabHankel0(r, x3, h, H, n) + uabK0(r, x3, h, H, n))
                    + (-2*r1*r2)/r**3 * (uabHankel1(r, x3, h, H, n) + uabK1(r, x3, h, H, n)
                    + r*uabK0(r, x3, h, H, n))), [int(n_s), int(n_f)]) + (2*r1*r2)/r**4 * uabSimple(r, x3, h, H))

def u13(r1, r2, x3, h, H, n_f, n_s=1):
    r = mp.sqrt(r1**2 + r2**2)
    return r1/r * mp.nsum(lambda n: ua3Hankel(r, x3, h, H, n), [int(n_s), int(n_f)])

def u31(r1, r2, x3, h, H, n_f, n_s=1):
    r = mp.sqrt(r1**2 + r2**2)
    return r1/r * mp.nsum(lambda n: u3aHankel(r, x3, h, H, n), [int(n_s), int(n_f)])
