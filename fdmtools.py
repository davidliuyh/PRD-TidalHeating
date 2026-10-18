# Dependencies
import warnings

import h5py as h5
import numpy as np
import scipy as sp
import scipy.optimize as scopt
import scipy.interpolate as scinterp
import scipy.integrate as scinteg
import scipy.special as scsp
import scipy.stats as scstat
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
plt.rc('font', size=18)

import pyshtools as pysh


# __all__ = ['rvir_calc']


# Constants
G = 4.30091727e-6               # Gravitational constant in kpc / M_sun * (km / s)^2
hbar = 6.582119569509067e-16    # Reduced Planck constant in eV * s
c_constant = 2.99792458e5       # Speed of light in km/s

# Conversions:
km_to_kpc = 3.240779289444365e-17       # Conversion from km to kpc
km_s_to_kpc_gyr = 1.022712165045695     # Conversion from km/s to kpc/Gyr

# Cosmology
Om = 0.2865                     # Omega matter
rho_crit = 133.36363195167573   # Critical background density (solMass / kpc^3)
rho_m = rho_crit * Om           # Background matter density at present
overdensity = 347.0             # Overdensity for spherical collapse
rho_vir = overdensity * rho_m   # Average density within the virial radius

class FDMTools:

    def __init__(self, m_a=8.1e-23):
        # Quantum Mechanics
        self.m_a = m_a                                          # FDM particle mass in eV / c^2
        self.h_over_m = hbar / (self.m_a / c_constant**2) * km_to_kpc     # Scaled unit (h_bar / m_a) in astrophysical units: (kpc * km / s)
    
    
    def t_freefall(self, r, M_enc):
        """
        Compute the freefall time at a given radius
        
        Parameters
        ----------
        M_enc : numeric
            Mass enclosed within the radius in solar masses
        r : numeric
            Radius in kpc
        
        Returns
        -------
        T_ff : numeric
            Freefall time in Gyr
        """
        G_Gyr = G * km_s_to_kpc_gyr**2.
        T_ff = np.sqrt((np.pi**2. * r**3.) / (8. * M_enc * G_Gyr))
        return T_ff
    
    
    def rvir_calc(self, M_halo):
        """
        Compute the virial radius of a halo
        
        Parameters
        ----------
        M_halo : numeric
            Halo mass in solar masses
        
        Returns
        -------
        r_vir : numeric
            Virial radius in kpc
        """
        r_vir = (M_halo / (4.0 / 3.0 * np.pi * rho_vir))**(1.0 / 3.0)
        return r_vir
    
    
    def dBw_estimator(self, rmin, rmax, rho, samples=20):
        """
        Estimate the de Broglie wavelength for a halo given its density profile
        
        Parameters
        ----------
        rmin : numeric
            Minimum radius in kpc
        rmax : numeric
            Maximum radius in kpc
        rho : lambda function or spline object
            Lambda function or cubic spline object that returns the value of the density as a function of radius
        
        Returns
        -------
        r_vir : numeric
            Virial radius in kpc
        """
        r_test = np.logspace(np.log10(rmin), np.log10(rmax), samples)
        dM = lambda r: 4. * np.pi * rho(r) * r**2
        v_c = np.zeros(samples)
    
        for i in range(samples):
            M_fit, err = scinteg.quad(dM, 0., r_test[i])
            v_c[i] = np.sqrt(G * M_fit / r_test[i])
    
        v_max = np.max(v_c)
        dBw = self.h_over_m / v_max
    
        return dBw
    
    
    def r_core(self, M_halo, powerlaw=1/3):
        """
        Compute the core radius of a FDM halo given the Schive et al. (2014) Core-Halo relation
        
        Parameters
        ----------
        M_halo : numeric
            Halo mass in solar masses
        
        Returns
        -------
        r_core : numeric
            Core radius in kpc
        """
        m22 = self.m_a / 1e-22
        r_c = 1.6 / m22 * (M_halo / 1e9)**(-powerlaw)
        return r_c
    
    
    def rho_core(self, r, r_c):
        """
        Compute the core density profile of a FDM halo given the Schive et al. (2014) Core-Halo relation
        
        Parameters
        ----------
        r : numpy array
            Array of radial distances
        M_halo : numeric
            Halo mass in solar masses
        r_c : numeric
            Core radius of the FDM halo in kpc
        
        Returns
        -------
        rho_c : numeric
            Density at radius r in solar masses per kpc^3
        """
        m23 = self.m_a / 1e-23
        rho_c = (1.9 * m23**(-2.) * r_c**(-4.)) / (1. + 0.091 * (r / r_c)**2.)**8.
        return rho_c * 1e9
    
    
    def phi_calc(self, r, rho):
        """
        Compute the gravitational potential at a radius given the density, using Poisson's equation
        
        Parameters
        ----------
        r : numpy array
            Array of radial distances
        rho : lambda function or spline object
            Lambda function or cubic spline object that returns the value of the density as a function of radius
        
        Returns
        -------
        phi : numpy array
            Values of the gravitational potential corresponding to the input radii in (km / s)^2
    
        """
        integrand1 = lambda r: rho(r) * r**2
        integrand2 = lambda r: rho(r) * r
    
        warnings.filterwarnings("ignore")
    
        integral1, err = scinteg.quad(integrand1, 0., r)
        integral2, err = scinteg.quad(integrand2, r, np.inf)
    
        return -4. * np.pi * G * (1. / r * integral1 + integral2)
    
    
    def initialize_potential(self, rho_in, core=False, r_max=None, powerlaw=1/3):
        """
        Initialize functions for calculating the potential and density as a function of radius
        
        Parameters
        ----------
        rho_in : function
            A Python function that returns the density in Solar Masses per kpc^3 at a given radius (in kpc)
        core : bool (optional, default: False)
            Superpose a soliton core based on a powerlaw Core-Halo relation
        powerlaw : numeric (default: 1/3)
            Powerlaw relation between the halo mass and the core mass. Default is 1/3, following the Schive et al. (2014) relation
        
        Returns
        -------
        phi_out : scipy ppoly object (cubic spline)
            Cubic spline object that returns the value of the potential in (km / s)^2 as a function of radius (in kpc)
        rho_out : scipy ppoly object (cubic spline)
            Cubic spline object that returns the value of the density in Solar Masses per kpc^3 as a function of radius
    
        """
        rspline = np.logspace(-3, 3, 256)
        rho_halo = rho_in(rspline)
    
        if core==False:
    
            rho_out = scinterp.CubicSpline(rspline, rho_halo)
    
            phispline = np.array([self.phi_calc(r, rho_in) for r in rspline])
            phi_out = scinterp.CubicSpline(rspline, phispline)
    
        elif core==True:
            dM = lambda r: 4. * np.pi * rho_in(r) * r**2
    
            if r_max is None:
                M_halo, err = scinteg.quad(dM, 0., np.inf)
            else:
                M_halo, err = scinteg.quad(dM, 0., r_max)
    
            print('Soliton for halo mass '+"{:.2e}".format(M_halo)+' Solar Masses')
    
            r_c = self.r_core(M_halo, powerlaw=powerlaw)
            print('Core radius = '+"{:.2e}".format(r_c)+' kpc')
            rho_c = self.rho_core(rspline, r_c)
    
            if np.any(rho_c > rho_halo)==False:
                print('Core density never reaches halo density; please input larger mass halo or different powerlaw.')
    
            transition = np.max(np.where(rho_c[rho_halo>rho_m] > rho_halo[rho_halo>rho_m])) + 1
    
            rho_combined = np.concatenate((rho_c[:transition], rho_halo[transition:]))
    
            rho_out = scinterp.CubicSpline(rspline, rho_combined)
            phispline = np.array([self.phi_calc(r, rho_out) for r in rspline])
            phi_out = scinterp.CubicSpline(rspline, phispline)
    
        dM = lambda r: 4. * np.pi * rho_out(r) * r**2
        if r_max is None:
            M_total, err = scinteg.quad(dM, 0., np.inf)
        else:
            M_total, err = scinteg.quad(dM, 0., r_max)
    
        return phi_out, rho_out, M_total
    
    
    def DF_invert(self, phi, rho):
        """
        Inversion formula to calculate the distribution function f(E) of a potential given an array of total energy
    
        Parameters
        ----------
        phi : scipy ppoly object (cubic spline)
            Cubic spline object that returns the value of the potential in (km / s)^2 as a function of radius (in kpc)
        rho : scipy ppoly object (cubic spline)
            Cubic spline object that returns the value of the density in Solar Masses per kpc^3 as a function of radius
        
        Returns
        -------
        f_E : scipy ppoly object (cubic spline)
            Cubic spline object that returns the value of the distribution function as a function of the total energy in (km / s)^2
    
        """
        r_spline = np.logspace(-3, 3, 256)
        phi_spline = phi(r_spline)
        rho_spline = rho(r_spline)
        idx = np.argsort(phi_spline)
    
        e_range = phi_spline[idx]
    
        rho_phi = scinterp.CubicSpline(phi_spline[idx], rho_spline[idx])
        drho_dphi = rho_phi.derivative()
    
        def integrand(phi_integ, e_prime):
            return 1. / np.sqrt(phi_integ - e_prime) * drho_dphi(phi_integ)
    
        prefactor = 1. / (2. * np.sqrt(2.) * np.pi**2)
    
        integ_spline = np.array([scinteg.quad(integrand, e_prime, e_range[-1], args=(e_prime,))[0] for e_prime in e_range[:-1]])
    
        integral = scinterp.CubicSpline(e_range[:-1], prefactor * integ_spline)
        f_E = integral.derivative()
    
        return f_E
    
    
    def DF_invert_radial(self, phi, rho):
        """
        Inversion formula to calculate the distribution function f(E) of a potential given an array of total energy
    
        Parameters
        ----------
        phi : scipy ppoly object (cubic spline)
            Cubic spline object that returns the value of the potential in (km / s)^2 as a function of radius (in kpc)
        rho : scipy ppoly object (cubic spline)
            Cubic spline object that returns the value of the density in Solar Masses per kpc^3 as a function of radius
        
        Returns
        -------
        f_E : scipy ppoly object (cubic spline)
            Cubic spline object that returns the value of the distribution function as a function of the total energy in (km / s)^2
    
        """
        r_spline = np.logspace(-3, 3, 1024)
        phi_spline = phi(r_spline)
        rho_spline = rho(r_spline)
        idx = np.argsort(phi_spline)
    
        e_range = phi_spline[idx]
    
        r_phi = scinterp.CubicSpline(phi_spline[idx], r_spline[idx])
    
        rho_phi = scinterp.CubicSpline(phi_spline[idx], rho_spline[idx])
    
        def integrand(phi_integ, e_prime):
            # return 1. / np.sqrt(phi_integ - e_prime) * rho_phi(phi_integ)
            return 1. / np.sqrt(phi_integ - e_prime) * rho_phi(phi_integ) * r_phi(phi_integ)**2
    
    
        integ_spline = np.array([scinteg.quad(integrand, e_prime, e_range[-1], args=(e_prime,))[0] for e_prime in e_range[:-1]])
    
        integral = scinterp.CubicSpline(e_range[:-1], integ_spline)
        deriv = integral.derivative()
    
        # prefactor = -r_phi(e_range[:-1])**2 / (np.sqrt(2.) * np.pi**2)
        prefactor = -1. / (np.sqrt(2.) * np.pi**2)
        deriv_spline = deriv(e_range[:-1])
    
        f_E = scinterp.CubicSpline(e_range[:-1], prefactor * deriv_spline)
    
        return f_E
    
    
    def g_E(self, phi):
        """
        Determination of the classical density of states (typically labeled g(E)) given an input potential 
    
        Parameters
        ----------
        phi : scipy ppoly object (cubic spline)
            Cubic spline object that returns the value of the potential in (km / s)^2 as a function of radius (in kpc)
    
        Returns
        -------
        g_E : scipy ppoly object (cubic spline)
            Cubic spline object that returns the value of the density of states as a function of the total energy in (km / s)^2
    
        """
        Emin = phi(0.) * (1. - 1e-6)
        Emax = phi(0.) * 1e-6
        Espline = np.linspace(Emin, Emax, 1024)
    
        def integrand(r, Einteg):
            return r**2 * np.sqrt(2. * (Einteg - phi(r)))
    
        g_Espline = np.array([(4. * np.pi)**2 * scinteg.quad(integrand, 0., phi.solve(E, extrapolate=False)[0], args=(E,))[0] for E in Espline])
    
        gE = scinterp.CubicSpline(Espline, g_Espline)
    
        return gE
    
    
    def dndE(self, E_nl, phi, bins=100, sparse_adjust=True):
        """
        Numerical calculation of the density of states (or states per energy bin dn / dE) as a function of total energy
    
        """
        m_shape = np.zeros(np.shape(E_nl)[1], dtype=int)
        for i in range(np.shape(E_nl)[1]):
            m_shape[i] = 2 * i + 1
        E_nl_repeat = np.repeat(E_nl, m_shape, axis=1)
        Eflat = E_nl_repeat.flatten()[E_nl_repeat.flatten()!=0]
        bindata, binedges = np.histogram(Eflat, bins=bins, range=(phi(0.), np.max(Eflat)))
    
        if sparse_adjust==False:
            dnde = np.zeros_like(Eflat)
            for i in range(int(len(Eflat))):
                dnde[i] = bindata[np.argmax(binedges>Eflat[i]) - 1]
            return Eflat, dnde
    
        elif sparse_adjust==True:
            bindata1 = np.zeros(len(bindata))
            indx=0
            for i in range(len(bindata)):
                if i==0:
                    continue
                if bindata[i-1] == 0 and bindata[i] != 0:
                    multiplier = i - indx
                    indx = i
                else:
                    multiplier = 1.
                bindata1[i] = bindata[i] / multiplier
    
            dnde = np.zeros_like(Eflat)
            for i in range(int(len(Eflat))):
                dnde[i] = bindata1[np.argmax(binedges>Eflat[i]) - 1] / (binedges[1] - binedges[0])
            return Eflat, dnde
    
    
    
    
    def valsvecs(self, r, step, l, phi, Emax, Rmax, verbose=False):
        """
        For a given value of l, a gravitational potential, and a maximum total energy, returns the eigenvalues and eigenvectors of the Schrodinger equation.
        Uses the finite difference method.
    
        """
        nsteps = len(r)
        Dr2_0 = -self.h_over_m**2 / (2 * step**2) * np.ones(nsteps) * (-2.)
        Dr2_1 = -self.h_over_m**2 / (2 * step**2) * np.ones(nsteps - 1)
        Vvec = self.h_over_m**2 * l * (l + 1) / (2. * r**2) + np.array([phi(ri) for ri in r])
            
        E_n, u_vecs_T = sp.linalg.eigh_tridiagonal(Dr2_0 + Vvec, Dr2_1, select='v', select_range=(-np.inf, Emax))
        u_vecs = u_vecs_T.T
        
        R_n = np.empty((1, len(E_n)), dtype=object)
    
        if l==0:
            E_n_diff = np.diff(E_n)
        
        for i in range(len(E_n)):
    
            R0 = u_vecs[i] / r
            A = 1. / np.sqrt(np.trapz(R0**2. * r**2., r))
            R1 = R0 * A
            R_n[0, i] = scinterp.CubicSpline(r, R1)
    
            Mass_in = np.trapz(R0[r<=Rmax]**2. * r[r<=Rmax]**2., r[r<=Rmax])
            Mass_out = np.trapz(R0[r>Rmax]**2. * r[r>Rmax]**2., r[r>Rmax])
    
            imax = i
    
            if Mass_out > Mass_in * 0.01:
                print('Mass_out greater than Mass_in -- should not happen if R_max is much greater than R_fit!')
                break
    
        E_n = E_n[:imax]
        R_n = R_n[:, :imax]
    
        if np.all(np.diff(E_n, n=2)[:-1] < 1e-8)==False:
            print('Warning! There may be missing and/or duplicated l='+str(l)+' eigenvalues; insufficient resolution')
            if verbose==True:
                print(E_n)
                print(np.diff(E_n, n=2)[:-1])
    
        return E_n, R_n
    
    
    
    def compute_radial_solutions(self, phi, rho, M_halo, r_max, r_fit, verbose=True):
        """
        Compute all eigenvalues and eigenvectors (for all l) for the Schrodinger equation
        """
        r_c = self.r_core(M_halo)
        r_dB = self.dBw_estimator(0.001, r_max, rho)
        r_resolve = np.min([r_c, r_dB])
    
        r_min = 0.002 * r_resolve
        step = 0.002 * r_resolve
        rsolve = np.arange(r_min, r_max, step)
    
        print('Sampling grid for eigenvalue problem includes '+str(len(rsolve))+' bins, out to r_max = '+str(round(r_max, 2))+' kpc')
        print('(returning only eigenmodes with energies below circular orbits at r_fit = '+str(round(r_fit, 2))+' kpc)')
    
        dM = lambda r: 4. * np.pi * rho(r) * r**2
        M_fit, err = scinteg.quad(dM, 0., r_fit)
        Ek = 0.5 * G * M_fit / r_fit
        Efit = phi(r_fit) + Ek
    
        E_nl_grid = []
        l = 0
        shape_n = 1
        total_modes_nl = 0
        total_modes_nlm = 0
        while shape_n>0:
            E_n, R_n = self.valsvecs(rsolve, step, l, phi, Efit, r_max, verbose=verbose)
            shape_n = len(E_n)
            if shape_n==0:
                break
            if l==0:
                R_nl_grid = np.copy(R_n.T)
                n_max = shape_n
                print('n_max = '+str(n_max))
    
            elif shape_n == n_max:
                R_nl_grid = np.hstack((R_nl_grid, R_n.T))
    
            else:
                R_n_long = np.concatenate((R_n, np.empty((1, n_max - shape_n), dtype=object)), axis=1)
                R_nl_grid = np.hstack((R_nl_grid, R_n_long.T))
    
            E_nl_array = np.zeros(n_max)
            E_nl_array[:shape_n] = E_n
    
            E_nl_grid.append(E_nl_array)
    
            # if verbose==True:
            print('l = '+str(l)+'; n_max = '+str(shape_n))
            total_modes_nl += shape_n
            total_modes_nlm += shape_n * (2 * l + 1)
            l+=1
        l_max = l
    
        print('n_max = '+str(n_max)+'; l_max = '+str(l_max)+'; distinct nl modes = '+str(total_modes_nl)+'; total modes = '+str(total_modes_nlm))
    
        return np.array(E_nl_grid).T, R_nl_grid
    
    
    
    def amp_from_fE(self, E_nl, fE, Type='Isotropic'):
        """
        Compute the eigenmode amplitudes a_nlm directly from an input distribution function
        """
        if Type=='Radial':
            factor = np.sqrt((2. * np.pi)**3. * self.h_over_m)
            E_nl[:, 1:] = 0.
        elif Type=='Isotropic':
            factor = np.sqrt((2. * np.pi * self.h_over_m)**3.)
        else:
            raise RuntimeError("Type must be either 'Isotropic' or 'Radial'")
    
        a_nlm = np.zeros_like(E_nl)
    
        a_nlm[E_nl!=0.] = np.nan_to_num(np.sqrt(fE(E_nl[E_nl!=0.]))) * factor
    
        return a_nlm
    
    
    def amp_from_schwarzschild_E(self, r, rho, E_nl, R_nl, bins=50, a_init=None, fE=None, max_iterations=2500, verbose=True):
        """
        Compute the eigenmode amplitudes a_nl such that (a_nl @ R_nl)**2 = the density profile (Schwarzschild method).
        Degenerate m modes are not accounted for.
        """
        den_target = rho(r)
        R_E2 = np.zeros((bins, len(r)))
    
        shape_nl = np.shape(E_nl)
        Emax = np.max(E_nl[E_nl!=0.])
        Emin = np.min(E_nl)
        Eedges = np.linspace(Emin, Emax, bins+1)
        Ebins = (Eedges[:-1] + Eedges[1:]) / 2.
    
        phi_ang = 0.
        theta_ang = 0.
        
        Y_lm_abs = np.abs(pysh.expand.spharm(shape_nl[1], theta_ang, phi_ang, normalization='ortho', degrees=False, kind='complex', csphase=-1))
        Y_l = (np.sum(Y_lm_abs[0, :, :], axis=1) + np.sum(Y_lm_abs[1, :, -shape_nl[1]:], axis=1))[:-1]
    
        for l in range(shape_nl[1]):
            for n in range(shape_nl[0]):
                if E_nl[n, l]==0.:
                    continue
                else:
                    indx = np.max(np.where(Eedges[:-1]<=E_nl[n, l]))
                    R_E2[indx] += (R_nl[n, l](r) * Y_l[l])**2
    
        if a_init==None:
            factor = np.sqrt((2. * np.pi * self.h_over_m)**3.)
            a_E = np.nan_to_num(np.sqrt(fE(Ebins))) * factor
            a_init = np.full_like(a_E, np.log10(np.mean(a_E)))
    
        dof = 0
        for i in range(len(Ebins)):
            if np.count_nonzero((E_nl >= Eedges[i]) & (E_nl < Eedges[i+1])):
                dof+=1
    
        print(str(dof)+' unique amplitudes')
    
        func_to_min = lambda amps: np.log10(1. / np.max(r) * np.trapz(((den_target - ((10**amps)**2) @ (R_E2))**2 / den_target**2), r))
    
        if verbose==True:
            global Nfeval
            Nfeval = 0
            def callbackF(Xi):
                global Nfeval
                if Nfeval%100==0:
                    print('{0:4d}   {1: 3.6f}'.format(Nfeval, func_to_min(Xi)))
                Nfeval += 1
            new_amps = scopt.minimize(func_to_min, a_init, callback=callbackF, options={'maxiter': max_iterations})
            a_E = np.abs(10**new_amps.x)
            print(new_amps.success)
            print('Optimization complete after '+str(new_amps.nit)+' iterations')
            print(new_amps.message)
        
        else:
            new_amps = scopt.minimize(func_to_min, a_init, options={'maxiter': max_iterations})
            a_E = np.abs(10**new_amps.x)
            print('Optimization complete after '+str(new_amps.nit)+' iterations')
    
        den_total = (a_E**2) @ (R_E2)
    
        a_nlm = np.zeros_like(E_nl)
    
        for n in range(shape_nl[0]):
            for l in range(shape_nl[1]):
                if E_nl[n, l] == 0.:
                    continue
                else:
                    indx = np.max(np.where(Eedges[:-1]<=E_nl[n, l]))
                    a_nlm[n, l] = a_E[indx]
    
        M_enc = np.trapz(4. * np.pi * r**2. * den_total, r)
    
        return a_nlm, den_total, M_enc
    
    
    def amp_from_schwarzschild(self, r, rho, E_nl, R_nl, a_nlm_init=None, Type='Isotropic', fE=None, max_iterations=2500, verbose=True):
        """
        Compute the eigenmode amplitudes a_nl such that (a_nl @ R_nl)**2 = the density profile (Schwarzschild method).
        Returns a_nl such that degenerate modes are accounted for.
        """
        den_target = rho(r)
        Eflat = E_nl.flatten()[E_nl.flatten()!=0]
        Rflat_func = R_nl.flatten()[E_nl.flatten()!=0]
        Rflat = np.zeros((len(Eflat), len(r)))
        for i in range(len(Rflat_func)):
            Rflat[i] = Rflat_func[i](r)
        
        func_to_min = lambda amps: np.log10(1. / np.max(r) * np.trapz(((den_target - ((10**amps)**2) @ (Rflat**2))**2 / den_target**2), r))
    
        if a_nlm_init is None:
            a_nlm_init = self.amp_from_fE(E_nl, fE, Type=Type)
            a_nl_init = self.amp_spher_to_rad(a_nlm_init)
            aflat_init = a_nl_init.flatten()[E_nl.flatten()!=0]
            a_init = np.full_like(aflat_init, np.log10(np.mean(aflat_init)))
        else:
            a_nl_init = self.amp_spher_to_rad(a_nlm_init)
            a_init = np.log10(a_nl_init.flatten()[E_nl.flatten()!=0])
    
        print(str(len(a_init))+' unique amplitudes')
    
        if verbose==True:
            global Nfeval
            Nfeval = 0
            def callbackF(Xi):
                global Nfeval
                if Nfeval%20==0:
                    print('{0:4d}   {1: 3.6f}'.format(Nfeval, func_to_min(Xi)))
                Nfeval += 1
            new_amps = scopt.minimize(func_to_min, a_init, method='BFGS', callback=callbackF, options={'maxiter': max_iterations})
            aflat = np.abs(10**new_amps.x)
            print(new_amps.success)
            print('Optimization complete after '+str(new_amps.nit)+' iterations')
            print(new_amps.message)
            print(new_amps.fun)
        
        else:
            new_amps = scopt.minimize(func_to_min, a_init, method='BFGS', options={'maxiter': max_iterations})
            aflat = np.abs(10**new_amps.x)
            print('Optimization complete after '+str(new_amps.nit)+' iterations')
    
        den_total = (aflat**2.) @ (Rflat**2.)
        a_nl = np.zeros_like(E_nl)
        a_nl[E_nl != 0] = aflat
        a_nlm = self.amp_rad_to_spher(a_nl)
    
        M_enc = np.trapz(4. * np.pi * r**2. * den_total, r)
    
        return a_nlm, den_total, M_enc
    
    
    def amp_rad_to_spher(self, a_nl):
        """
        Convert a_nl to a_nlm (to account for degenerate m modes and enable creation of 3D spherical halos)
        """
        dim = np.shape(a_nl)
        a_scale = np.sqrt(4. * np.pi / np.tile(np.arange(1., 2. * dim[1] + 1., 2.), (dim[0], 1)))
        # a_scale = np.sqrt(4. * np.pi / np.tile(np.arange(1., 2. * dim[1] + 1., 2.), dim[0]).reshape(dim))
        a_nlm = a_nl * a_scale
        return a_nlm
    
    
    def amp_spher_to_rad(self, a_nlm):
        """
        Convert a_nlm to a_nl (to consolidate degenerate m modes and enable efficient calculation of the 1D density profile)
        """
        dim = np.shape(a_nlm)
        a_scale = np.sqrt(4. * np.pi / np.tile(np.arange(1., 2. * dim[1] + 1., 2.), (dim[0], 1)))
        a_nl = a_nlm / a_scale
        return a_nl
    
    
    def den_from_amps(self, r, a_nlm, R_nl):
        """
        Calculate time-averaged density profile and total enclosed mass from amplitudes and eigenvectors for any input radius (no quantum interference)
        """
        shape_nl = np.shape(R_nl)
        den_total = np.zeros(len(r))
    
        phi_ang = 0.
        theta_ang = 0.
        Y_lm_abs = np.abs(pysh.expand.spharm(shape_nl[1], theta_ang, phi_ang, normalization='ortho', degrees=False, kind='complex', csphase=-1))
        Y_l = np.zeros(shape_nl[1])
        for l in range(shape_nl[1]):
            for m in range(0, l+1):
                Y_l[l] += Y_lm_abs[0, l, m]
            for m in range(-l, 0):
                Y_l[l] += Y_lm_abs[1, l, -m]
    
        for n in range(shape_nl[0]):
            for l in range(shape_nl[1]):
                if a_nlm[n, l]==0.:
                    continue
                else:
                    den_total += (a_nlm[n, l] * R_nl[n, l](r) * Y_l[l])**2.
    
        M_enc = np.trapz(4. * np.pi * r**2. * den_total, r)
    
        return den_total, M_enc
    
    
    def random_phase(self, nmax, lmax, seed=0):
        """
        Output a random phase for each eigenmode (n, l, and m) given lmax and nmax
        """
        np.random.seed(seed)
        phase_nlm = np.zeros((nmax, lmax, 2 * lmax + 1))
        for n in range(nmax):
            for l in range(lmax):
                for m in range(-l, l + 1):
                    phase_nlm[n, l, m] = np.random.rand() * 2. * np.pi
        
        return phase_nlm
    
    
    def solve_amps(self, rfit, phi, rho, fE, E_range, E_nl, R_nl, method, bins=50, s_iter=500, verbose=True):
        """
        Solve for best fit amplitudes
        """
        if method=='DF_isotropic':
            a_nlm = self.amp_from_fE(E_nl, fE, Type='Isotropic')
            den, M_enc = self.den_from_amps(rfit, a_nlm, R_nl)
    
            factor = np.sqrt((2. * np.pi * self.h_over_m)**3.)
            fE_compare = np.nan_to_num(np.sqrt(fE(E_range))) * factor
    
        elif method=='DF_radial':
            gE = self.DF_invert_radial(phi, rho)
            a_nlm = self.amp_from_fE(E_nl, gE, Type='Radial')
            den, M_enc = self.den_from_amps(rfit, a_nlm, R_nl)
    
            factor = np.sqrt((2. * np.pi)**3. * self.h_over_m)
            fE_compare = np.nan_to_num(np.sqrt(gE(E_range))) * factor
    
        elif method=='Isotropic':
            a_nlm, den, M_enc = self.amp_from_schwarzschild_E(rfit, rho, E_nl, R_nl, fE=fE, bins=bins, max_iterations=s_iter, verbose=True)
    
            factor = np.sqrt((2. * np.pi * self.h_over_m)**3.)
            fE_compare = np.nan_to_num(np.sqrt(fE(E_range))) * factor
    
        elif method=='Radial':
            E_nl[:, 1:] = 0.
            R_nl[:, 1:] = None
            gE = self.DF_invert_radial(phi, rho)
            a_nlm, den, M_enc = self.amp_from_schwarzschild(rfit, rho, E_nl, R_nl, Type='Radial', fE=gE, max_iterations=s_iter, verbose=True)
    
            factor = np.sqrt((2. * np.pi)**3. * self.h_over_m)
            fE_compare = np.nan_to_num(np.sqrt(gE(E_range))) * factor
    
        elif method=='Unconstrained':
            a_nlm, den, M_enc = self.amp_from_schwarzschild(rfit, rho, E_nl, R_nl, fE=fE, max_iterations=s_iter, verbose=True)
    
            factor = np.sqrt((2. * np.pi * self.h_over_m)**3.)
            fE_compare = np.nan_to_num(np.sqrt(fE(E_range))) * factor
            
        else:
            raise RuntimeError("Must specify method (DF_isotropic, DF_radial, Isotropic, Radial, or Unconstrained)")
    
        return a_nlm, den, M_enc, fE_compare
    
    
    def plot_iteration(self, rfit, rho_in, den_list, E_nl, a_nlm, E_range, fE_compare):
        """
        Plot the most recent iteration of the Schwarzschild solver
        """
        iterations = len(den_list)
    
        fig, f_axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 6), constrained_layout=True, sharey=False)
        f_axes[0].loglog(rfit, rho_in(rfit), 'b-', label='Target')
        for j in range(iterations - 1):
            f_axes[0].loglog(rfit, den_list[j+1](rfit), label='Iteration #'+str(j+1))
        f_axes[0].set_xlim(0.01, 100)
        f_axes[0].set_xlabel(r'$r$ [kpc]')
        f_axes[0].set_ylim(100, 10**(round(np.log10(rho_in(0.1)), 0) + 1))
        f_axes[0].set_ylabel(r'$\rho$ [$M_\odot$ / kpc$^3$]')
        f_axes[0].legend()
     
        f_axes[1].semilogy(E_nl, a_nlm, 'b+')
        f_axes[1].semilogy(E_nl[0, 0], a_nlm[0, 0], 'b+', label=r'$a_{nlm}$')
        f_axes[1].semilogy(E_range, fE_compare, 'k--', label=r'$f(E)$')
        f_axes[1].set_xlabel(r'$E$ [(km / s)$^2$]')
        f_axes[1].set_ylim(1, 10**(round(np.log10(np.max(a_nlm)), 0) + 1))
        f_axes[1].set_ylabel('Amplitude (unnormalized)')
        f_axes[1].legend()
        plt.show()
        plt.close()
    
        return
        
    
    def solve_amps_itr(self, r_max, r_fit, phi_in, rho_in, M_goal, fE, rsolve_max=None, rfit_man=None, converge_target=-3., maxiter=5, method='Unconstrained', ground_state=True, bins=50, schwarz_iter=500, verbose=True, plot_progress=True, seed=0):
        """
        Solve for best fit amplitudes by iterating several times
        """
    
        r_c = self.r_core(M_goal)
        r_dB = self.dBw_estimator(0.001, r_max, rho_in)
        r_resolve = np.min([r_c, r_dB])
    
        r_min = 0.002 * r_resolve
    
        step = 0.002 * r_resolve
    
        if rfit_man is not None:
            rfit = rfit_man
        else:
            rfit = np.arange(r_min, r_fit, step)
    
        if rsolve_max==None:
            rsolve_max = r_max
    
        r_interp = np.arange(r_min, rsolve_max * 1.2, step)
    
        M_enc = 4. * np.pi * np.trapz(rfit**2. * rho_in(rfit), rfit)
        print('Enclosed mass in rfit_max = 10^'+str(np.log10(M_enc)))
    
        den_list = []
    
        rho = rho_in
        phi = phi_in
        den_list.append(rho)
        
        for i in range(maxiter):
            print('Iteration #'+str(i+1))
            print('Solving for eigenmodes...')
            E_nl, R_nl = self.compute_radial_solutions(phi, rho, M_goal, rsolve_max, r_fit,verbose=False)
    
            if ground_state==False:
                E_nl[0, 0] = 0.
                R_nl[0, 0] = None
    
            E_range = np.linspace(0.99, 0.01) * phi(0)
    
            print('Solving for amplitudes...')
    
            a_nlm, den, M_enc, fE_compare = self.solve_amps(rfit, phi, rho, fE, E_range, E_nl, R_nl, method, bins=bins, s_iter=schwarz_iter, verbose=verbose)
    
            den_long, M_enc = self.den_from_amps(r_interp, a_nlm, R_nl)
    
            print('Updated enclosed mass in rfit = 10^'+str(np.log10(M_enc)))
    
            rho_interp = scinterp.interp1d(r_interp, den_long, bounds_error=False, fill_value=(den_long[0], 0.), assume_sorted=True)
    
            print('Solving for new potential...')
            phi_new, rho_new, M = self.initialize_potential(rho_interp)
    
            den_list.append(rho_new)
    
            rho_diff = (rho(rfit) - rho_new(rfit))**2 / (rho(rfit)**2)
            converge_crit = np.log10(1. / np.max(rfit) * np.trapz(rho_diff, rfit))
            print('Convergence Criterion after Iteration #'+str(i+1)+': D = 10^'+str(converge_crit))
            phi = phi_new
            rho = rho_new
    
            if plot_progress==True:
                self.plot_iteration(rfit, rho_in, den_list, E_nl, a_nlm, E_range, fE_compare)
    
            if converge_crit < converge_target and i > 0:
                print('Convergence criterion met!')
                break
    
        phase_nlm = self.random_phase(np.shape(E_nl)[0], np.shape(E_nl)[1], seed)
    
        print('Completed!')
    
        return E_nl, R_nl, a_nlm, phase_nlm, rho, den_list
    
    
    def amp_to_clm(self, aphase_n, RatShell, lmax):
        """
        Convert amplitude and radial eigenfunction into clm spectrum for SHTools
        """
        anlm = aphase_n * RatShell[:, None]
        anlm1 = anlm[:, :lmax+1]
        anlm2 = np.roll(np.flip(anlm[:, lmax:], axis=1), 1)
        power = np.zeros((2, lmax, lmax+1), dtype = complex)
        power[0] = anlm1
        power[1] = anlm2
        
        return power
    
    
    def build_halo(self, rshells, E_nl, R_nl, a_nlm, phase_nlm, lgrid=None):
        """
        Build a 3D halo using spherical harmonic transformations
        """
        nmax = np.shape(E_nl)[0]
        lmax = np.shape(E_nl)[1]
        
        if lgrid is None:
            if lmax<60:
                lgrid = 80
            else:
                lgrid = lmax + lmax // 3
    
        aphase = (np.cos(phase_nlm) + 1j * np.sin(phase_nlm)) * a_nlm[:,:,None]
        
        clm_setup = pysh.SHCoeffs.from_zeros(lmax=lgrid, kind='complex')
        grid_setup = clm_setup.expand(lmax=lgrid)
        
        grid3d_zeros = np.zeros((len(rshells),) + np.shape(grid_setup.data))
        Rs = np.copy(grid3d_zeros) + rshells[:, None, None]
        thetas = np.copy(grid3d_zeros) + ((-grid_setup.lats() + 90.) / 180. * np.pi)[None, :, None]
        phis = np.copy(grid3d_zeros) + ((grid_setup.lons() - 180) / 180. * np.pi)[None, None, :]
        grid3d = np.zeros((len(rshells),) + np.shape(grid_setup.data), dtype='complex')
        
        RatShell = np.zeros(lmax)
            
        for i in range(len(rshells)):
            for n in range(nmax):
                for l in range(lmax):
                    if E_nl[n, l]!=0.:
                        RatShell[l] = R_nl[n, l](rshells[i])
                
                power = self.amp_to_clm(aphase[n], RatShell, lmax)
                
                clm = pysh.SHCoeffs.from_array(power, normalization='ortho')
                grid = clm.expand(lmax=lgrid)
                
                grid3d[i] += grid.data
        
        return grid3d, Rs, thetas, phis
    
    
    
    def calculate_potential_grid(self, rshells, E_nl, R_nl, a_nlm, phase_nlm, lgrid=None):
    
        nmax = np.shape(E_nl)[0]
        lmax = np.shape(E_nl)[1]
        
        if lgrid is None:
            if lmax<60:
                lgrid = 80
            else:
                lgrid = lmax + lmax // 3
    
        aphase = (np.cos(phase_nlm) + 1j * np.sin(phase_nlm)) * a_nlm[:,:,None]
        
        clm_setup = pysh.SHCoeffs.from_zeros(lmax=lgrid, kind='complex')
        grid_setup = clm_setup.expand(lmax=lgrid)
    
        rho_grid = []
    
        grid3d_zeros = np.zeros((len(rshells),) + np.shape(grid_setup.data))
        Rs = np.copy(grid3d_zeros) + rshells[:, None, None]
        thetas = np.copy(grid3d_zeros) + ((-grid_setup.lats() + 90.) / 180. * np.pi)[None, :, None]
        phis = np.copy(grid3d_zeros) + ((grid_setup.lons() - 180) / 180. * np.pi)[None, None, :]
    
        RatShell = np.zeros(lmax)
    
        for i in range(len(rshells)):
            rho_sqrt = grid_setup
            for n in range(nmax):
                for l in range(lmax):
                    if E_nl[n, l]!=0.:
                        RatShell[l] = R_nl[n, l](rshells[i])
    
                power = self.amp_to_clm(aphase[n], RatShell, lmax)
    
                clm = pysh.SHCoeffs.from_array(power, normalization='ortho')
                grid = clm.expand(lmax=lgrid)
    
                rho_sqrt += grid
            rho_grid.append(np.abs(rho_sqrt)**2)
    
        rho_lm_shells = []
        rho_lm_shells_coeffs = np.zeros(np.shape(rho_grid[0].expand().coeffs) + (len(rshells),))
        for i in range(len(rshells)):
            rho_lm_shells.append(rho_grid[i].expand())
            rho_lm_shells_coeffs[:, :, :, i] = rho_lm_shells[i].coeffs
    
        ls = np.indices(np.shape(rho_lm_shells[0].coeffs))[1]
    
        integrands1_range = np.zeros_like(rho_lm_shells_coeffs)
        integrands2_range = np.zeros_like(rho_lm_shells_coeffs)
    
        for i in range(len(rshells)):
            integrands1_range[:, :, :, i] = rshells[i]**(1 - ls) * rho_lm_shells_coeffs[:, :, :, i]
            integrands2_range[:, :, :, i] = rshells[i]**(2 + ls) * rho_lm_shells_coeffs[:, :, :, i]
    
        Phi_lm = []
        Phi_lm_coeffs = np.zeros(np.shape(rho_grid[0].expand().coeffs) + (len(rshells),))
        for i in range(len(rshells)):
            integrals1 = scinteg.simpson(integrands1_range[:, :, :, i:], rshells[i:])
            integrals2 = scinteg.simpson(integrands2_range[:, :, :, :i+1], rshells[:i+1])
            Phi_lm_coeffs[:, :, :, i] = -(4 * np.pi * G) / (2 * ls + 1) * (rshells[i]**ls * integrals1 + rshells[i]**(-(ls + 1)) * integrals2)
            Phi_lm.append(pysh.SHCoeffs.from_array(Phi_lm_coeffs[:, :, :, i], normalization='4pi'))
    
    
        Phi_grid3d = np.zeros((len(rshells),) + np.shape(grid_setup.data))
    
        for i in range(len(rshells)):
            Phi_grid3d[i] = Phi_lm[i].expand(lmax=lgrid).data
    
        return Phi_grid3d, Rs, thetas, phis
    
    
    
    def plot_slices(self, grid3d, Rs, thetas, phis, res, rmax, rshells):
        """
        Plot xy, xz, and yz slices for a fdm halo grid
        """
        Rs_flat = Rs.flatten()
        phis_flat = phis.flatten()
        thetas_flat = thetas.flatten()
    
        xs_flat = Rs_flat * np.sin(thetas_flat) * np.cos(phis_flat)
        ys_flat = Rs_flat * np.sin(thetas_flat) * np.sin(phis_flat)
        zs_flat = Rs_flat * np.cos(thetas_flat)
    
        grid_flat = grid3d.flatten()
    
        boxedge = rmax
    
        axis1 = np.linspace(-boxedge, boxedge, res)
        axis2 = np.linspace(-boxedge, boxedge, res)
        Xxy,Yxy,Zxy = np.meshgrid(axis1, axis2, 0., indexing='ij')
        Xxz,Yxz,Zxz = np.meshgrid(axis1, 0. ,axis2, indexing='ij')
        Xyz,Yyz,Zyz = np.meshgrid(0., axis1, axis2, indexing='ij')
    
        grid_xy = sp.interpolate.griddata((xs_flat, ys_flat, zs_flat), grid_flat, (Xxy, Yxy, Zxy), method='nearest', fill_value = 0.)[:, :, 0]
        grid_xz = sp.interpolate.griddata((xs_flat, ys_flat, zs_flat), grid_flat, (Xxz, Yxz, Zxz), method='nearest', fill_value = 0.)[:, 0, :]
        grid_yz = sp.interpolate.griddata((xs_flat, ys_flat, zs_flat), grid_flat, (Xyz, Yyz, Zyz), method='nearest', fill_value = 0.)[0, :, :]
        
        rshell_mean = np.mean(grid3d, axis=(1, 2))
        bar_max = np.log10(np.max(grid_xy))
        bar_min = np.log10(np.min(rshell_mean[rshells < boxedge]))
    
        plt.loglog(rshells, rshell_mean, 'k-')
        plt.xlim([np.min(rshells), boxedge])
        plt.ylim([np.min(rshell_mean[rshells < boxedge]) / 10., rshell_mean[0] * 10.])
        plt.show()
        
        fig, f_axes = plt.subplots(ncols=3, figsize=(24, 8), constrained_layout=True, sharex=False, sharey=False)
    
        f_axes[0].imshow(np.log10(grid_xy).T,vmin=bar_min,vmax=bar_max,aspect='equal',cmap='PuOr_r',extent=[-boxedge,boxedge,-boxedge,boxedge], origin='lower')
        f_axes[1].imshow(np.log10(grid_xz).T,vmin=bar_min,vmax=bar_max,aspect='equal',cmap='PuOr_r',extent=[-boxedge,boxedge,-boxedge,boxedge], origin='lower')
        ax = f_axes[2]
        pcm = f_axes[2].imshow(np.log10(grid_yz).T,vmin=bar_min,vmax=bar_max,aspect='equal',cmap='PuOr_r',extent=[-boxedge,boxedge,-boxedge,boxedge], origin='lower')
    
        f_axes[0].set_xlabel('$x$ [kpc]')
        f_axes[1].set_xlabel('$x$ [kpc]')
        f_axes[2].set_xlabel('$y$ [kpc]')
        f_axes[0].set_ylabel('$y$ [kpc]')
        f_axes[1].set_ylabel('$z$ [kpc]')
        f_axes[2].set_ylabel('$z$ [kpc]')
        
        fig.colorbar(pcm, ax=ax, label=r'$\rho$ [$M_\odot$ / kpc$^3$]')
        
        plt.show()
        
        return grid_xy, grid_xz, grid_yz, rshells, rshell_mean
    
    
    def halo_to_cube(self, grid3d, Rs, thetas, phis, res, rmax):
        """
        Project halo to a cube in cartesian coordinates
        """
        Rs_flat = Rs.flatten()
        phis_flat = phis.flatten()
        thetas_flat = thetas.flatten()
    
        xs_flat = Rs_flat * np.sin(thetas_flat) * np.cos(phis_flat)
        ys_flat = Rs_flat * np.sin(thetas_flat) * np.sin(phis_flat)
    
        boxedge = rmax
    
        zs_flat = Rs_flat * np.cos(thetas_flat)
    
        grid_flat = grid3d.flatten()
        
        x = np.linspace(-boxedge, boxedge, res)
        y = np.linspace(-boxedge, boxedge, res)
        z = np.linspace(-boxedge, boxedge, res)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
        psi = sp.interpolate.griddata((xs_flat, ys_flat, zs_flat), grid_flat, (X, Y, Z), method='nearest', fill_value = 0.)
        
        return psi
    
    
    def cube_to_enzo_input(self, psi, res, boxedge, enzo_dens=1.8788e-29, enzo_time=2.519445e17, absorption_boundary=False, zero_momentum=False):
    
        dens_gal_to_cgs = 6.7679e-32
        dens_to_enzo = dens_gal_to_cgs / enzo_dens
        kpc_to_cm = 3.0857e21
    
        rank = 3
        dim = res
    
        attr = {'Component_Rank':1, 'Component_Size':dim**rank, 'Dimensions':[dim]*rank, 'Rank':rank, 'TopGridDims':[dim]*rank, 'TopGridEnd':[dim]*rank, 'TopGridStart':[0]*rank}
    
        # renormalize so mean is 1
        dens = np.zeros([dim]*rank)
        xdim,ydim,zdim = dens.shape
        repsi = np.zeros([dim]*rank)
        impsi = np.zeros([dim]*rank)
    
        # set
        if (zero_momentum == True):
            side = np.linspace(0,res,res,endpoint=False)
            x,y,z = np.meshgrid(side,side,side,indexing='ij')
            
            # compute velocity
            vx0 = np.imag(np.gradient(psi,axis=0)*np.conjugate(psi))
            vy0 = np.imag(np.gradient(psi,axis=1)*np.conjugate(psi))
            vz0 = np.imag(np.gradient(psi,axis=2)*np.conjugate(psi))
            # center of mass velocity
            vbarx = np.sum(vx0)/np.sum(np.absolute(psi)**2)
            vbary = np.sum(vy0)/np.sum(np.absolute(psi)**2)
            vbarz = np.sum(vz0)/np.sum(np.absolute(psi)**2)
    
            print("initial COM velocity",vbarx,vbary,vbarz)
    
            # shift the phase to zero the COM velocity
            psi = psi*np.exp(-1j*(vbarx*x + vbary*y + vbarz*z))
        
            # check and diagnostic
            vx0 = np.imag(np.gradient(psi,axis=0)*np.conjugate(psi))
            vy0 = np.imag(np.gradient(psi,axis=1)*np.conjugate(psi))
            vz0 = np.imag(np.gradient(psi,axis=2)*np.conjugate(psi))
            vbarx = np.sum(vx0)/np.sum(np.absolute(psi)**2)
            vbary = np.sum(vy0)/np.sum(np.absolute(psi)**2)
            vbarz = np.sum(vz0)/np.sum(np.absolute(psi)**2)
            print("final COM velocity",vbarx,vbary,vbarz)
        
        repsi = np.real(psi)*np.sqrt(dens_to_enzo)
        impsi = np.imag(psi)*np.sqrt(dens_to_enzo)
        dens = np.absolute(psi)**2.0*dens_to_enzo
                    
        if (absorption_boundary == True):
            side = np.linspace(0,dim,dim,endpoint=False)/dim-0.5
            x,y,z = np.meshgrid(side,side,side,indexing='ij')
            r = np.sqrt(x**2+y**2+z**2)
    
            dens = 1e2*np.arctan((r-0.48)*1e3)
            dens[r<0.48]=0.
        else:
            dens = np.absolute(psi)**2.0*dens_to_enzo
    
        # write out new density to new file
        f1 = h5.File('./GridDensity', 'w')
        new_dens = f1.create_dataset('GridDensity', data=dens)
        for a in attr:
            new_dens.attrs.create(a, attr[a])
        f1.close()
    
        # write out to new file
        f1 = h5.File('./GridRePsi', 'w')
        new_dens = f1.create_dataset('GridRePsi', data=repsi)
        for a in attr:
            new_dens.attrs.create(a, attr[a])
        f1.close()
    
        f1 = h5.File('./GridImPsi', 'w')
        new_dens = f1.create_dataset('GridImPsi', data=impsi)
        for a in attr:
            new_dens.attrs.create(a, attr[a])
        f1.close()
    
        enzo_len = 2 * boxedge * kpc_to_cm
        M_tot = np.sum(np.absolute(psi)**2) * (2 * boxedge / (res - 1))**3
        t_ff = self.t_freefall(boxedge, M_tot)
    
        f2 = open('halo_info.txt', 'w')
        f2.write('Boxside = '+str(boxedge*2)+' kpc\n')
        f2.write('Enzo LengthUnits = '+str(enzo_len)+'\n')
        f2.write('Total Mass = 10^'+str(np.log10(M_tot))+' solMass\n')
        f2.write('Freefall time = '+str(t_ff)+' Gyr ~ '+str(t_ff/8)+' Enzo TimeUnits\n')
        f2.close()
    
        return
    
        
    def build_eigenmode_cubes(self, states, rsolve, R_nl, rcube, res, m_modes=None):
    
        x = np.linspace(-rcube, rcube, res)
        y = np.linspace(-rcube, rcube, res)
        z = np.linspace(-rcube, rcube, res)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
        Rad = np.sqrt(X**2.0+Y**2.0+Z**2.0)
        phi = np.arctan2(Y,X)
        theta = np.arccos(Z/Rad)
    
        if m_modes is None:
            gridnum = len(states)
            print('No m_modes files provided; calculating only m=0 modes')
        else:
            gridnum = len(m_modes) - np.count_nonzero(m_modes) + np.sum(2 * states[m_modes][:, 1] + 1)
    
        psi = np.zeros((gridnum, res, res, res), dtype=np.complex64)
        indx = np.zeros((gridnum, 3))
    
        j = 0
    
        for i in range(len(states)):
            n = states[i, 0]
            l = states[i, 1]
            if m_modes is None:
                m = 0
                indx[i] = [n, l, m]
                RatRad = R_nl[i](Rad)
                spharm = scsp.sph_harm(m, l, phi, theta)
                psi[i] = (RatRad * spharm).astype(np.complex64)
    
            else:
                if m_modes[i]==False:
                    m = 0
                    indx[j] = [n, l, m]
                    RatRad = R_nl[i](Rad)
                    spharm = scsp.sph_harm(m, l, phi, theta)
                    psi[j] = (RatRad * spharm).astype(np.complex64)
                    j+=1
                else:
                    for m in range(-l, l+1):
                        indx[j] = [n, l, m]
                        RatRad = R_nl[i](Rad)
                        spharm = scsp.sph_harm(m, l, phi, theta)
                        psi[j] = (RatRad * spharm).astype(np.complex64)
    
                        j += 1
    
        return psi, indx
    
    
    def halo_decompose(self, psi, boxsize, boxedge, Mh0, n_max=None, l_max=None, den=None):
        """
        Decompose a halo into eigenmode amplitudes
        """
        if den==None:
            print("Function currently incomplete -- must include input density for now")
            return
        else:
            phi, rho = self.initialize_potential(den, core=False)
            rsolve, E_nl, R_nl, shape_nl = self.compute_radial_solutions(phi, r_max=boxedge * 1.1, verbose=True)
    
        x = np.linspace(-boxedge,boxedge,boxsize)
        y = np.linspace(-boxedge,boxedge,boxsize)
        z = np.linspace(-boxedge,boxedge,boxsize)
        X,Y,Z = np.meshgrid(x,y,z)
        Rad = np.sqrt(X**2.0+Y**2.0+Z**2.0)
        phi = np.arctan2(Y,X)
        theta = np.arccos(Z/Rad)
    
        binside = boxedge * 2.0 / boxsize
    
        if n_max==None:
            n_max = np.shape(E_nl)[0]
        if l_max==None:
            l_max = np.shape(E_nl)[1]
    
        A_nlm = np.zeros([n_max, l_max, 2 * l_max + 1])
        psi = np.zeros_like(Rad, dtype=complex)
        RatRad = np.zeros((n_max, np.shape(Rad)[0], np.shape(Rad)[1], np.shape(Rad)[2]))
    
        for l in range(l_max):
            print(l)
    
            for n in range(n_max):
                if E_nl[n, l] == 0.:
                    continue
                else:
                    RatRad[n] = R_nl[n, l](Rad)
    
            for m in range(0, l+1):
                spharm_interp = scsp.sph_harm(m, l, phi, theta) 
                for n in range(n_max):
                    if E_nl[n, l] == 0.:
                        continue
                    else:
                        RatRad0 = RatRad[n]
                        if m==0:
                            psi0 = RatRad0 * spharm_interp
                            A_nlm[n, l, m] = np.sum(psi * psi0 * binside**3.)
                        else:
                            psi0 = RatRad0 * spharm_interp
                            A_nlm[n, l, m] = np.sum(psi * psi0 * binside**3.)
                            psi0 = RatRad0 * (-1)**m * np.conjugate(spharm_interp)
                            A_nlm[n, l, -m] = np.sum(psi * psi0 * binside**3.)
        return A_nlm
    
    
    def evolve(self, rshell, E_nl, rsolve, R_nl, a_nlm, phase_nlm, t):
        # input: t in Gyr
        phase_k = -E_nl / self.h_over_m * km_s_to_kpc_gyr * t
        phase = phase_nlm + phase_k[:,:,None] 
        return self.build_halo(rshell, E_nl, rsolve, R_nl, a_nlm, phase)
    
    
    def find_vortices(self, psi,threshold=8):
        # velocity field
        vx = np.zeros(psi.shape)
        vy = np.zeros(psi.shape)
        vz = np.zeros(psi.shape)
    
        # vorticity field
        wx = np.zeros(psi.shape)
        wy = np.zeros(psi.shape)
        wz = np.zeros(psi.shape)
        
        psi = psi/np.absolute(psi)
    
        # compute velocity vector
        vx[1:-1,:,:] = np.imag((psi[2:,:,:]-psi[0:-2,:,:])*np.conjugate(psi[1:-1,:,:]))
        vy[:,1:-1,:] = np.imag((psi[:,2:,:]-psi[:,0:-2,:])*np.conjugate(psi[:,1:-1,:]))
        vz[:,:,1:-1] = np.imag((psi[:,:,2:]-psi[:,:,0:-2])*np.conjugate(psi[:,:,1:-1]))
        
        # compute vorticity vector
        wx[2:-2,2:-2,2:-2] = vz[2:-2,3:-1,2:-2] - vz[2:-2,1:-3,2:-2] - vy[2:-2,2:-2,3:-1] + vy[2:-2,2:-2,1:-3]
        wy[2:-2,2:-2,2:-2] = vx[2:-2,2:-2,3:-1] - vx[2:-2,2:-2,1:-3] - vz[3:-1,2:-2,2:-2] + vz[1:-3,2:-2,2:-2]
        wz[2:-2,2:-2,2:-2] = vy[3:-1,2:-2,2:-2] - vy[1:-3,2:-2,2:-2] - vx[2:-2,3:-1,2:-2] + vx[2:-2,1:-3,2:-2]
        
        curlv = abs(wx)**2 + abs(wy)**2 + abs(wz)**2
                            
        xs,ys,zs = np.where(curlv>threshold)
        return xs,ys,zs
    
    
    def plot_vortices(self, psi):
        # find points on the vortex lines
        xs,ys,zs = self.find_vortices(psi)
        # glue points on the same vortex line together
        data = np.array(list(zip(xs,ys,zs)))
        clustering = DBSCAN(eps=2.5, min_samples=2).fit(data)
    
        cmap=plt.get_cmap('tab10')
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
    
    
        ax.scatter(xs, ys, zs, s=0.2,c=cmap(np.mod(clustering.labels_,10)),rasterized=True)
    
        ax.view_init(30, 30)
    
        ax.set_xlabel(r'$x$', fontsize=24)
        ax.set_ylabel(r'$y$', fontsize=24)
        ax.set_zlabel(r'$z$', fontsize=24)
    
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

