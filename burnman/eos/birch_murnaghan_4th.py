from __future__ import absolute_import
# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU GPL v2 or later.


import scipy.optimize as opt
from . import equation_of_state as eos
import warnings


def bulk_modulus(volume, params):
    """
    compute the bulk modulus as per the third order
    birch-murnaghan equation of state.  Returns bulk
    modulus in the same units as the reference bulk
    modulus.  Pressure must be in :math:`[Pa]`.
    """
    B1 = (params['K_0']*params['Kprime_prime_0'])+((params['Kprime_0']-4.)*(params['Kprime_0']-5.))+(59./9.)
    B2 = (3.*params['K_0']*params['Kprime_prime_0'])+((params['Kprime_0']-4.)*((3.*params['Kprime_0'])-13.))+(129./9.)
    B3 = (3.*params['K_0']*params['Kprime_prime_0'])+((params['Kprime_0']-4.)*((3.*params['Kprime_0'])-11.))+(105./9.)
    B4 = (params['K_0']*params['Kprime_prime_0'])+((params['Kprime_0']-4.)*(params['Kprime_0']-3.))+(35./9.)

    x = params['V_0']/volume

    K = (9./16.)*params['K_0']*( (-5.*B1/3.)*pow(x,-5./3.)+(7.*B2/3.)*pow(x,-7./3.)-3*B3*pow(x,-3.)+(11.*B4/3.)*pow(x,-11./3.))
    return K

def volume(pressure,params):

    func = lambda x: birch_murnaghan(x/params['V_0'], params) - pressure
    V = opt.brentq(func, .1*params['V_0'], 1.5*params['V_0'])
    return V

def birch_murnaghan(x, params):
    B1 = (params['K_0']*params['Kprime_prime_0'])+((params['Kprime_0']-4.)*(params['Kprime_0']-5.))+(59./9.)
    B2 = (3.*params['K_0']*params['Kprime_prime_0'])+((params['Kprime_0']-4.)*((3.*params['Kprime_0'])-13.))+(129./9.)
    B3 = (3.*params['K_0']*params['Kprime_prime_0'])+((params['Kprime_0']-4.)*((3.*params['Kprime_0'])-11.))+(105./9.)
    B4 = (params['K_0']*params['Kprime_prime_0'])+((params['Kprime_0']-4.)*(params['Kprime_0']-3.))+(35./9.)

    return (9.*params['K_0']/16.)*((-B1*pow(x, -5./3.)+(B2*pow(x, -7./3.))) - (B3*pow(x, -3.))+(B4*pow(x, -11./3.)))

def shear_modulus_third_order(volume, params):
    """
    Get the birch murnaghan shear modulus at a reference temperature, for a
    given volume.  Returns shear modulus in :math:`[Pa]` (the same units as in
    params['G_0']).  This uses a third order finite strain expansion
    """

    x = params['V_0']/volume
    f = 0.5*(pow(x, 2./3.) - 1.0)
    G = pow((1. + 2*f), 5./2.)*(params['G_0']+(3.*params['K_0']*params['Gprime_0'] - 5.*params['G_0'])*f + (6.*params['K_0']*params['Gprime_0']-24.*params['K_0']-14.*params['G_0']+9./2. * params['K_0']*params['Kprime_0'])*f*f)
    return G


class BM4(eos.EquationOfState):
    """
    Base class for the isothermal Birch Murnaghan equation of state.  This is third order in strain, and
    has no temperature dependence.  However, the shear modulus is sometimes fit to a second order 
    function, so if this is the case, you should use that.  For more see :class:`burnman.birch_murnaghan.BM2` and :class:`burnman.birch_murnaghan.BM3`.
    """
    def volume(self,pressure, temperature, params):
        """
        Returns volume :math:`[m^3]` as a function of pressure :math:`[Pa]`.
        """
        return volume(pressure,params)

    def pressure(self, temperature, volume, params):
        return birch_murnaghan((volume/params['V_0'], params))

    def isothermal_bulk_modulus(self,pressure,temperature, volume, params):
        """
        Returns isothermal bulk modulus :math:`K_T` :math:`[Pa]` as a function of pressure :math:`[Pa]`,
        temperature :math:`[K]` and volume :math:`[m^3]`. 
        """
        return bulk_modulus(volume,params)
    def adiabatic_bulk_modulus(self,pressure, temperature, volume, params):
        """
        Returns adiabatic bulk modulus :math:`K_s` of the mineral. :math:`[Pa]`.
        """
        return bulk_modulus(volume,params)

    def shear_modulus(self,pressure, temperature, volume, params):
        """
        Returns shear modulus :math:`G` of the mineral. :math:`[Pa]`
        """
        return shear_modulus_third_order(volume,params)
    def heat_capacity_v(self,pressure, temperature, volume, params):
        """
        Since this equation of state does not contain temperature effects, simply return a very large number. :math:`[J/K/mol]`
        """
        return 1.e99

    def heat_capacity_p(self,pressure, temperature, volume, params):
        """
        Since this equation of state does not contain temperature effects, simply return a very large number. :math:`[J/K/mol]`
        """
        return 1.e99

    def thermal_expansivity(self,pressure, temperature, volume, params):
        """
        Since this equation of state does not contain temperature effects, simply return zero. :math:`[1/K]`
        """
        return 0.

    def grueneisen_parameter(self,pressure,temperature,volume,params):
        """
        Since this equation of state does not contain temperature effects, simply return zero. :math:`[unitless]`
        """
        return 0.

    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """
     
        if 'P_0' not in params:
            params['P_0'] = 0.

        # If G and Gprime are not included this is presumably deliberate,
        # as we can model density and bulk modulus just fine without them,
        # so just add them to the dictionary as nans
        if 'G_0' not in params:
            params['G_0'] = float('nan')
        if 'Gprime_0' not in params:
            params['Gprime_0'] = float('nan')
  
        # Check that all the required keys are in the dictionary
        expected_keys = ['V_0', 'K_0', 'Kprime_0']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)
        
        # Finally, check that the values are reasonable.
        if params['P_0'] < 0.:
            warnings.warn( 'Unusual value for P_0', stacklevel=2 )
        if params['V_0'] < 1.e-7 or params['V_0'] > 1.e-3:
            warnings.warn( 'Unusual value for V_0', stacklevel=2 )
        if params['K_0'] < 1.e9 or params['K_0'] > 1.e13:
            warnings.warn( 'Unusual value for K_0' , stacklevel=2)
        if params['Kprime_0'] < 0. or params['Kprime_0'] > 10.:
            warnings.warn( 'Unusual value for Kprime_0', stacklevel=2 )
