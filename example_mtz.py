# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
    Examples of MTZ applications
    To do:
    - get composites in composites, so ringwoodite can split in periclase and perovskite
    - incorporate latent heat from phase transitions
    - put in self-consistent conversion to depth
    
    
    """

import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals

if __name__ == "__main__":
    
    hot=plt.get_cmap('hot')
    
    method = 'slb3'

    
    amount_perovskite = 0.8
    rock = minerals.SLB_MTZ.mg_olivine(0.1, amount_perovskite)
    rock.set_method(method)
    
    
    #seismic model for comparison:
    # pick from .prem() .slow() .fast() (see burnman/seismic.py)
    seismic_model = burnman.seismic.PREM()
    
    depths = np.linspace(100e3, 2800e3, 30)
    p, seis_rho, seis_vp, seis_vs, seis_vphi = seismic_model.evaluate_all_at(depths)
    
    # Now we get an array of temperatures at which will be used for computing
    # the seismic properties of the rock.
    T = np.linspace(1200,3500,30)
    
    print "pressures:\n", p
    print "temperatures:\n", T
    
    # turn grid into array:
    tarray=np.tile(T,len(p))
    parray=np.repeat(p,len(T))
    
    density, vp, vs, vphi, K, G = burnman.velocities_from_rock(rock, parray, tarray)
    
    mat_vs = np.reshape(vs,[len(p),len(T)]);
    mat_vp = np.reshape(vp,[len(p),len(T)]);
    mat_vphi = np.reshape(vphi,[len(p),len(T)]);
    mat_density = np.reshape(density,[len(p),len(T)]);
    
    fig = plt.figure(1)
    
    X,Y = np.meshgrid(p/1e9, T)
    plt.subplot(2,2,1)
    plt.pcolor(X,Y,mat_vs.transpose()/1.e3,shading='interp')#,color=plt.cm.jet)
    plt.colorbar()
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("Temperature")
    plt.autoscale(tight=True)
    plt.title("Vs (km/s)")

    plt.subplot(2,2,2)
    plt.pcolor(X,Y,mat_vp.transpose()/1.e3,shading='interp')#,color=plt.cm.jet)
    plt.colorbar()
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("Temperature")
    plt.autoscale(tight=True)
    plt.title("Vp (km/s)")
    
    plt.subplot(2,2,3)
    plt.pcolor(X,Y,mat_vphi.transpose()/1.e3,shading='interp')#,color=plt.cm.jet)
    plt.colorbar()
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("Temperature")
    plt.autoscale(tight=True)
    plt.title("Vphi (km/s)")

    plt.subplot(2,2,4)
    plt.pcolor(X,Y,mat_density.transpose(),shading='interp')#,color=plt.cm.jet)
    plt.colorbar()
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("Temperature")
    plt.autoscale(tight=True)
    plt.title("density")


    fig = plt.figure(2)
    depths = np.linspace(100e3, 2800e3, 80)

    p, seis_rho, seis_vp, seis_vs, seis_vphi = seismic_model.evaluate_all_at(depths)
    T = burnman.geotherm.brown_shankland(p)
    density, vp, vs, vphi, K, G = burnman.velocities_from_rock(rock, p, T)

    plt.plot(p/1.e9,density/1.e3,color='b',linestyle='-',marker='o', markerfacecolor='b',markersize=4, label='model')
    plt.plot(p/1.e9,seis_rho/1.e3,color='k',linestyle='-',markerfacecolor='k',markersize=4, label='prem')
    plt.xlim(min(p)/1.e9,max(p)/1.e9)
    plt.title("density")
    plt.legend(loc='lower right')

    plt.show()
