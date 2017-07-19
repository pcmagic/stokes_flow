import numpy as np
from scipy.io import loadmat

class detail():
    def __init__(self):
        self._threshold = 0
        self._k = np.zeros([0])
        self._n = np.zeros([0])
        self._xn = np.zeros([0])
        self._yn = np.zeros([0])
        self._DmyD_xn = np.zeros([0])
        self._DmyD_yn = np.zeros([0])
        self._xn_k0 = np.zeros([0])
        self._yn_k0 = np.zeros([0])
        self._DmyD_xn_k0 = np.zeros([0])
        self._DmyD_yn_k0 = np.zeros([0])
        self._psi_xn1 = np.zeros([0])
        self._psi_xn2 = np.zeros([0])
        self._psi_xn3 = np.zeros([0])
        self._pi_xn1 = np.zeros([0])
        self._pi_xn2 = np.zeros([0])
        self._pi_xn3 = np.zeros([0])
        self._omega_xn1 = np.zeros([0])
        self._omega_xn2 = np.zeros([0])
        self._omega_xn3 = np.zeros([0])
        self._psi_yn1 = np.zeros([0])
        self._psi_yn2 = np.zeros([0])
        self._psi_yn3 = np.zeros([0])
        self._pi_yn1 = np.zeros([0])
        self._pi_yn2 = np.zeros([0])
        self._pi_yn3 = np.zeros([0])
        self._omega_yn1 = np.zeros([0])
        self._omega_yn2 = np.zeros([0])
        self._omega_yn3 = np.zeros([0])
        self._psi_xn1_k0 = np.zeros([0])
        self._psi_xn3_k0 = np.zeros([0])
        self._pi_xn1_k0 = np.zeros([0])
        self._pi_xn3_k0 = np.zeros([0])
        self._omega_xn1_k0 = np.zeros([0])
        self._omega_xn3_k0 = np.zeros([0])
        self._psi_yn2_k0 = np.zeros([0])
        self._pi_yn2_k0 = np.zeros([0])
        self._omega_yn2_k0 = np.zeros([0])

    def get_xyk(self):
        threshold = self._threshold
        kmax = np.round(threshold - 2)
        nmax = np.round(threshold / 2)
        k_use, n_use = np.meshgrid(np.arange(-kmax, kmax+1), np.arange(1, nmax+1))
        INDEX = (np.abs(k_use) + 2 * n_use) <= threshold
        INDEX[kmax,:] = 0
        k_use = k_use[INDEX]
        n_use = n_use[INDEX]
        mat_contents = loadmat('xn.mat')
        xn = mat_contents['xn']
        mat_contents = loadmat('yn.mat')
        yn = mat_contents['yn']
        xn_use = np.vstack((xn[kmax:1:-1, 1: nmax+1], xn[1: kmax + 1, 1: nmax+1]))
        yn_use =  np.vstack((yn[kmax:1-1:, 1: nmax+1], yn[1: kmax + 1, 1: nmax+1]))
        xn_use = xn_use[INDEX]
        yn_use = yn_use[INDEX]
        xn_k0 = xn[0, 1:nmax]
        yn_k0 = yn[0, 1:nmax]

        self._k = k_use
        self._n = n_use
        self._xn = xn_use
        self._yn = yn_use
        self._xn_k0 = xn_k0
        self._yn_k0 = yn_k0
        return True

    def psi1(self, k, s, b):
        myans = (1/16).*pi.^(-2).*(s.^2.*((besseli((-2)+k,s)+besseli(k,s)).*besseli(1+k,s)+besseli((-1)+k,s).*(besseli(k,s)+besseli(2+k,s))).*(besseli((-1)+k,b.*s).*besselk((-1)+k,s)+(-2).*b.*besseli(k,b.*s).*besselk(k,s)+besseli(1+k,b.*s).*besselk(1+k,s))+(-1).*(s.*besseli((-1)+k,s)+(-1).*((-1)+k).*besseli(k,s)).*(besseli(1+k,s).*(b.*s.*(besseli((-2)+k,b.*s)+3.*besseli(k,b.*s)).*besselk((-1)+k,s)+besseli((-1)+k,b.*s).*((-2).*s.*besselk((-2)+k,s)+(-2).*(1+k).*besselk((-1)+k,s))+(-2).*s.*besseli(1+k,b.*s).*besselk(k,s))+2.*besseli((-1)+k,s).*((-1).*s.*(besseli((-1)+k,b.*s)+besseli(1+k,b.*s)).*besselk(k,s)+2.*(b.*s.*besseli(k,b.*s)+(-1).*(2+k).*besseli(1+k,b.*s)).*besselk(1+k,s))))
        return myans
    def psi2(self, k, s, b):
        myans = (1/16).*pi.^(-2).*(s.^2.*((besseli((-2)+k,s)+besseli(k,s)).*besseli(1+k,s)+besseli((-1)+k,s).*(besseli(k,s)+besseli(2+k,s))).*(besseli((-1)+k,b.*s).*besselk((-1)+k,s)+(-1).*besseli(1+k,b.*s).*besselk(1+k,s))+(-4).*b.^(-1).*(s.*besseli((-1)+k,s)+(-1).*((-1)+k).*besseli(k,s)).*(b.*((-2)+k).*besseli((-1)+k,b.*s).*besseli(1+k,s).*besselk((-1)+k,s)+(-1).*k.*besseli(k,b.*s).*besseli(1+k,s).*besselk(k,s)+besseli((-1)+k,s).*((-1).*k.*besseli(k,b.*s).*besselk(k,s)+b.*(2+k).*besseli(1+k,b.*s).*besselk(1+k,s))));
        return myans
    def psi3(self, k, s, b):
        myans = (1/8).*pi.^(-2).*s.*(((besseli((-2)+k,s)+besseli(k,s)).*besseli(1+k,s)+besseli((-1)+k,s).*(besseli(k,s)+besseli(2+k,s))).*((-1).*b.*s.*besseli((-1)+k,b.*s).*besselk(k,s)+besseli(k,b.*s).*(s.*besselk((-1)+k,s)+2.*((-1)+k).*besselk(k,s)))+(-2).*(s.*besseli((-1)+k,s)+(-1).*((-1)+k).*besseli(k,s)).*(b.*besseli((-1)+k,b.*s).*besseli(1+k,s).*besselk((-1)+k,s)+(-1).*besseli(k,b.*s).*besseli(1+k,s).*besselk(k,s)+besseli((-1)+k,s).*((-1).*besseli(k,b.*s).*besselk(k,s)+b.*besseli(1+k,b.*s).*besselk(1+k,s))));
        return myans
    def pi1(self, k, s, b):
        myans = (1/16).*pi.^(-2).*(besseli(k,s).*besseli(1+k,s).*(b.*s.*(besseli((-2)+k,b.*s)+3.*besseli(k,b.*s)).*besselk((-1)+k,s)+besseli((-1)+k,b.*s).*((-2).*s.*besselk((-2)+k,s)+(-2).*(1+k).*besselk((-1)+k,s))+(-2).*s.*besseli(1+k,b.*s).*besselk(k,s))+(-2).*besseli((-1)+k,s).*(s.*besseli((-1)+k,b.*s).*(2.*besseli(1+k,s).*besselk((-1)+k,s)+besseli(k,s).*besselk(k,s))+(-2).*b.*s.*besseli(k,b.*s).*(2.*besseli(1+k,s).*besselk(k,s)+besseli(k,s).*besselk(1+k,s))+besseli(1+k,b.*s).*(2.*s.*besseli(1+k,s).*besselk(1+k,s)+besseli(k,s).*(s.*besselk(k,s)+2.*(2+k).*besselk(1+k,s)))));
        return myans
    def pi2(self, k, s, b):
        myans = (1/4).*b.^(-1).*pi.^(-2).*(besseli(k,s).*besseli(1+k,s).*(b.*((-2)+k).*besseli((-1)+k,b.*s).*besselk((-1)+k,s)+(-1).*k.*besseli(k,b.*s).*besselk(k,s))+besseli((-1)+k,s).*((-1).*b.*s.*besseli((-1)+k,b.*s).*besseli(1+k,s).*besselk((-1)+k,s)+b.*s.*besseli(1+k,s).*besseli(1+k,b.*s).*besselk(1+k,s)+besseli(k,s).*((-1).*k.*besseli(k,b.*s).*besselk(k,s)+b.*(2+k).*besseli(1+k,b.*s).*besselk(1+k,s))));
        return myans
    def pi3(self, k, s, b):
        myans = (1/4).*pi.^(-2).*((-1).*s.*besseli(k,s).*besseli(k,b.*s).*besseli(1+k,s).*besselk(k,s)+b.*s.*besseli((-1)+k,b.*s).*besseli(1+k,s).*(besseli(k,s).*besselk((-1)+k,s)+2.*besseli((-1)+k,s).*besselk(k,s))+besseli((-1)+k,s).*((-1).*besseli(k,b.*s).*(s.*besseli(k,s).*besselk(k,s)+2.*besseli(1+k,s).*(s.*besselk((-1)+k,s)+2.*((-1)+k).*besselk(k,s)))+b.*s.*besseli(k,s).*besseli(1+k,b.*s).*besselk(1+k,s)));
        return myans
    def omega1(self, k, s, b):
        myans = (1/16).*pi.^(-2).*s.^(-1).*(s.^2.*besseli((-1)+k,s).^2.*((-1).*b.*s.*besseli((-2)+k,b.*s).*besselk((-1)+k,s)+(-3).*b.*s.*besseli(k,b.*s).*besselk((-1)+k,s)+(-8).*b.*k.*besseli(k,b.*s).*besselk(k,s)+2.*besseli((-1)+k,b.*s).*(s.*besselk((-2)+k,s)+(1+3.*k).*besselk((-1)+k,s)+(-1).*s.*besselk(k,s))+4.*b.*s.*besseli(k,b.*s).*besselk(1+k,s)+(-8).*besseli(1+k,b.*s).*besselk(1+k,s))+(-2).*s.*besseli((-1)+k,s).*besseli(k,s).*((-1).*b.*((-1)+k).*s.*besseli((-2)+k,b.*s).*besselk((-1)+k,s)+3.*b.*s.*besseli(k,b.*s).*besselk((-1)+k,s)+(-3).*b.*k.*s.*besseli(k,b.*s).*besselk((-1)+k,s)+(-8).*b.*k.^2.*besseli(k,b.*s).*besselk(k,s)+2.*besseli((-1)+k,b.*s).*(((-1)+k).*s.*besselk((-2)+k,s)+((-1)+3.*k.^2).*besselk((-1)+k,s)+(-1).*((-1)+k).*s.*besselk(k,s))+(-4).*b.*s.*besseli(k,b.*s).*besselk(1+k,s)+4.*b.*k.*s.*besseli(k,b.*s).*besselk(1+k,s)+8.*besseli(1+k,b.*s).*besselk(1+k,s)+(-4).*k.*besseli(1+k,b.*s).*besselk(1+k,s))+besseli(k,s).^2.*((-2).*besseli((-1)+k,b.*s).*((4.*k.*s+s.^3).*besselk((-2)+k,s)+(4.*k+4.*k.^2+s.^2+3.*k.*s.^2).*besselk((-1)+k,s)+(-1).*s.^3.*besselk(k,s))+s.*(b.*(4.*k+s.^2).*besseli((-2)+k,b.*s).*besselk((-1)+k,s)+8.*besseli(1+k,b.*s).*((-1).*k.*besselk(k,s)+s.*besselk(1+k,s))+b.*besseli(k,b.*s).*(3.*(4.*k+s.^2).*besselk((-1)+k,s)+(-4).*s.*((-2).*k.*besselk(k,s)+s.*besselk(1+k,s))))));
        return myans
    def omega2(self, k, s, b):
        myans = (1/2).*b.^(-1).*pi.^(-2).*s.^(-1).*((-1).*b.*s.^2.*besseli((-1)+k,s).^2.*(besseli((-1)+k,b.*s).*besselk((-1)+k,s)+besseli(1+k,b.*s).*besselk(1+k,s))+b.*s.*besseli((-1)+k,s).*besseli(k,s).*(((-2)+3.*k).*besseli((-1)+k,b.*s).*besselk((-1)+k,s)+((-2)+k).*besseli(1+k,b.*s).*besselk(1+k,s))+besseli(k,s).^2.*(b.*(4.*k+(-2).*k.^2+s.^2).*besseli((-1)+k,b.*s).*besselk((-1)+k,s)+2.*k.^2.*besseli(k,b.*s).*besselk(k,s)+b.*s.^2.*besseli(1+k,b.*s).*besselk(1+k,s)));
        return myans
    def omega3(self, k, s, b):
        myans = (1/4).*pi.^(-2).*s.^(-1).*(s.*besseli(k,s).^2.*((-2).*k.*besseli(k,b.*s).*(s.*besselk((-1)+k,s)+2.*k.*besselk(k,s))+b.*besseli((-1)+k,b.*s).*((4.*k+s.^2).*besselk((-1)+k,s)+2.*k.*s.*besselk(k,s))+(-1).*b.*s.^2.*besseli(1+k,b.*s).*besselk(1+k,s))+s.*besseli((-1)+k,s).^2.*(2.*k.*besseli(k,b.*s).*(s.*besselk((-1)+k,s)+2.*((-1)+k).*besselk(k,s))+(-1).*b.*s.*besseli((-1)+k,b.*s).*(s.*besselk((-1)+k,s)+2.*k.*besselk(k,s))+b.*s.^2.*besseli(1+k,b.*s).*besselk(1+k,s))+2.*besseli((-1)+k,s).*besseli(k,s).*((-2).*k.^2.*besseli(k,b.*s).*(s.*besselk((-1)+k,s)+2.*((-1)+k).*besselk(k,s))+b.*s.*besseli((-1)+k,b.*s).*(((-1)+k).*s.*besselk((-1)+k,s)+2.*k.^2.*besselk(k,s))+(-1).*b.*((-1)+k).*s.^2.*besseli(1+k,b.*s).*besselk(1+k,s)));
        return myans

    def DmyD(self, k, s):
        myans = 2.*s.^(-2).*besseli(k,s).*((-1).*s.*((-4)+k.^2+s.^2).*besseli((-1)+k,s).^2+2.*((-2)+k).*(k.*(2+k)+s.^2).*besseli((-1)+k,s).*besseli(k,s)+s.*(k.*(4+k)+s.^2).*besseli(k,s).^2);
        return myans

    def solve_uR1k0(self):
        myans = (-2).*exp(1).^((-1).*z.*imag(xn)).*pi.*imag(DmyD.^(-1).*exp(1).^(sqrt(-1).*z.*real(xn)).*(pi1.*R.*xn.*besseli(0,R.*xn)+((-1).*pi1+psi1).*besseli(1,R.*xn)));
        return myans
    def solve_uR2k0(self):
        myans = 0
        return myans
    def solve_uR3k0(self):
        myans = 2.*exp(1).^((-1).*z.*imag(xn)).*pi.*real(DmyD.^(-1).*exp(1).^(sqrt(-1).*z.*real(xn)).*(pi3.*R.*xn.*besseli(0,R.*xn)+((-1).*pi3+psi3).*besseli(1,R.*xn)));
        return myans

    def solve_uz1k0(self):
        myans = (-2).*exp(1).^((-1).*z.*imag(xn)).*pi.*real(DmyD.^(-1).*exp(1).^(sqrt(-1).*z.*real(xn)).*((pi1+psi1).*besseli(0,R.*xn)+pi1.*R.*xn.*besseli(1,R.*xn)));
        return myans
    def solve_uz2k0(self):
        myans = 0
        return myans
    def solve_uz3k0(self):
        myans = (-2).*exp(1).^((-1).*z.*imag(xn)).*pi.*imag(DmyD.^(-1).*exp(1).^(sqrt(-1).*z.*real(xn)).*((pi3+psi3).*besseli(0,R.*xn)+pi3.*R.*xn.*besseli(1,R.*xn)));
        return myans

    def solve_uPhi1k0(self):
        myans = 0
        return myans
    def solve_uPhi2k0(self):
        myans = exp(1).^((-1).*z.*imag(yn)).*pi.*imag(DmyD.^(-1).*omega2.*besseli(1,R.*yn));
        return myans
    def solve_uPhi3k0(self):
        myans = 0
        return myans

    def solve_uPhi1(self):
        pass