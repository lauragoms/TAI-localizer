import sys
import numpy as np
import kwant
params = dict(Lx = 5,
         Ly = 5,
         Delta = 4.,
         B = 1, 
         A = 1,
         mu = 0)


sigma_0 = np.eye(2)
sigma_x = np.array(
    [[0, 1],
     [1, 0]])
sigma_y = np.array(
    [[0, -1j],
     [1j, 0]])
sigma_z = np.array(
    [[1, 0],
     [0, -1]])


def system_2D_BHZ(Lx,Ly,params=params):
    """
    Returns a 2D BHZ model system with given parameters.

    Parameters
    ----------
    params : dict
        Dictionary containing the model parameters

    Returns
    -------
    sys : kwant.builder.Builder
    """

    lat = kwant.lattice.square(norbs = 4)


    def onsite(site):
        return (params['Delta'] - 4*params['B'])* np.kron(sigma_z,sigma_0) + params['mu']*np.kron(sigma_0,sigma_0) # peru's code is -1 because of ws, wp = -1

    def hop_x(site0, site1):
        return (params['B']*np.kron(sigma_z,sigma_0) + params['A']/(2j) * np.kron(sigma_x,sigma_z))

    def	hop_y(site0, site1):
        return (params['B']*np.kron(sigma_z,sigma_0) - params['A']/(2j) * np.kron(sigma_y,sigma_0))

                                     
    sys = kwant.Builder()

    sys[(lat(x, y) for x in range(-Lx,Lx) for y in range(-Ly,Ly))] = onsite        
    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_x
    sys[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_y

    return sys,lat


def lead_BHZ(Ly,lat,params=params):


    def onsite(site):
        return (params['Delta'] - 4*params['B'])* np.kron(sigma_z,sigma_0) + params['mu']*np.kron(sigma_0,sigma_0) # peru's code is -1 because of ws, wp = -1
   
    def hop_x(site0, site1):
        return (params['B']*np.kron(sigma_z,sigma_0) + params['A']/(2j) * np.kron(sigma_x,sigma_z))

    def	hop_y(site0, site1):
        return (params['B']*np.kron(sigma_z,sigma_0) - params['A']/(2j) * np.kron(sigma_y,sigma_0))


    sym = kwant.TranslationalSymmetry(lat.vec((1,0)))
    lead = kwant.Builder(sym)
    
    lead[(lat(0, y) for y in range(-Ly,Ly))] = onsite        
    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_x
    sys[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_y

    return lead

def BHZ_with_leads(Lx,Ly,params=params):
    
    syst, lat= system_2D_BHZ(Lx, Ly, params=params)
    lead_plus  = lead_BHZ(Ly, lat, params=params)

    lead_minus = lead_plus.reversed()

    syst.attach_lead(lead_plus)          # to layer k=0
    syst.attach_lead(lead_minus) 

    return syst.finalized(), lead_plus.finalized(), lead_minus.finalized()

def TR_op(len_sites):
    return np.kron(np.kron(np.identity(len_sites),sigma_0),sigma_y)