
import numpy.linalg as npl
from numpy.linalg import norm
import kwant
import numpy as np
import math 
from numpy import linspace, array, dot, kron
import scipy
from fractions import Fraction
from math import exp, cos, pi, sin, sqrt
import tinyarray
from importlib import reload 
from itertools import product
from scipy import spatial
import copy 
import scipy.sparse as sp
import time,sys

def update_progress(progress, decimalpoints=0):
    """ Make an interactive progress bar as described on:
    https://stackoverflow.com/questions/3160699/python-progress-bar
    """
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + 
            "-"*(barLength-block), round(progress*100, decimalpoints), status)
    sys.stdout.write(text)
    sys.stdout.flush()  


'''Crystalline Lattice Definition'''

tau_0 = sigma_0 = s_0 = np.array([[1, 0], [0, 1]])
tau_x = sigma_x = s_x = np.array([[0, 1], [1, 0]])
tau_y = sigma_y = s_y = np.array([[0, -1j], [1j, 0]])

tau_z = sigma_z = s_z = np.array([[1, 0], [0, -1]])

a1=np.array([1,0,0])
a2=np.array([0,1,0])
a3=np.array([0,0,1])
c=a1+a2+a3

params=dict(pbc=False,
            R=1.01,
            MJ=2.3,
            lambdaJ=1,
            BJ=0.,
            break_PH=0.05,
            norbs=4,
            dis_onsite=0,
            seed_onsite=0,
            pbc_x=False,pbc_y=False,pbc_z=False, 
            mu_leads=0.7,
            mu=0)


    

def onsite_term(params=params):
    MJ, lambdaJ, BJ, break_PH, W = params['MJ'], params['lambdaJ'], params['BJ'], params['break_PH'], params['dis_onsite']
    mu = params['mu']
    disorder = np.random.uniform(-W/2, W/2)
    return (
        break_PH * kron(tau_0, sigma_0) +
        MJ * kron(tau_z, sigma_0) +
        kron(tau_0, (BJ/sqrt(3)) * (sigma_x + sigma_y + sigma_z)) +
        disorder * np.eye(4) + mu * kron(tau_0, sigma_0)
    )


def hopping_x(site1, site2, params):
    return kron(tau_z, sigma_0)*(1/2) - 1j * params['lambdaJ'] * kron(tau_x, sigma_x)*(1/2)

def hopping_y(site1, site2, params):
    return kron(tau_z, sigma_0)*(1/2) - 1j * params['lambdaJ'] * kron(tau_x, sigma_y)*(1/2)

def hopping_z(site1, site2, params):
    return kron(tau_z, sigma_0)*(1/2) - 1j * params['lambdaJ'] * kron(tau_x, sigma_z)*(1/2)


def crystalline_3D(L, params=params):

    """
    3D cubic TI/HOTI with onsite disorder

    Parameters
    ----------
    L : int
        Linear size of the cubic system (L x L x L)
    params : dict
        Dictionary of model parameters

    Returns
    ----------
    model : kwant.Builder
        Kwant Builder object
    """
    lat = kwant.lattice.general([a1,a2,a3], norbs=4)
    model = kwant.Builder()
    pbc_x,pbc_y,pbc_z =  params['pbc_x'],params['pbc_y'], params['pbc_z']

    if pbc_x or pbc_y or pbc_z:
        print('Periodic Boundary Conditions enabled for', end=' ')

        if pbc_x: print('x', end=' ')
        if pbc_y: print('y', end=' ')
        if pbc_z: print('z', end=' ')
        print()

    if params['dis_onsite']!=0:
        print('Disordered system with strength',params['dis_onsite'],'with seed',params['seed_onsite'])

    np.random.seed(int(params['seed_onsite']))
    # Onsite
    for x in range(L):
        for y in range(L):
            for z in range(L):
                model[lat(x, y, z)] = onsite_term(params)


    def add_hopping(a, b, hop_fn):
        model[a, b] = lambda s1, s2, p=params: hop_fn(s1, s2, p)

    # Hoppings
    for x in range(L):
        for y in range(L):
            for z in range(L):
                # +x
                if x + 1 < L:
                    add_hopping(lat(x,y,z), lat(x+1,y,z), hopping_x)
                elif params.get('pbc_x', False):
                    add_hopping(lat(x,y,z), lat(0,y,z), hopping_x)

                # +y
                if y + 1 < L:
                    add_hopping(lat(x,y,z), lat(x,y+1,z), hopping_y)
                elif params.get('pbc_y', False):
                    add_hopping(lat(x,y,z), lat(x,0,z), hopping_y)

                # +z
                if z + 1 < L:
                    add_hopping(lat(x,y,z), lat(x,y,z+1), hopping_z)
                elif params.get('pbc_z', False):
                    add_hopping(lat(x,y,z), lat(x,y,0), hopping_z)
    return model

def lead_in_z(L,params):
    # Translational symmetry along +z
    sym = kwant.TranslationalSymmetry((0, 0, 1))

    lat = kwant.lattice.general(
        [a1,a2,a3], 
        norbs=4
    )
    pbc_x,pbc_y,pbc_z =  params['pbc_x'],params['pbc_y'], params['pbc_z']

    if pbc_x or pbc_y or pbc_z:
        print('Leads with Periodic Boundary Conditions enabled for', end=' ')

        if pbc_x: print('x', end=' ')
        if pbc_y: print('y', end=' ')
        if pbc_z: print('z', end=' ')
        print()

    lead = kwant.Builder(sym)
    lead_params = params.copy()

    lead_params['dis_onsite'] = 0
    lead_params['mu'] = params['mu_leads']

    # Add sites only in the *unit cell layer* z=0
    for x in range(L):
        for y in range(L):
            lead[lat(x, y, 0)] = onsite_term(lead_params)

    # Add hoppings within layer
    for x in range(L):
        for y in range(L):
            # +x
            if x + 1 < L:
                lead[lat(x,y,0), lat(x+1,y,0)] = lambda s1,s2,p=lead_params: hopping_x(s1,s2,p)
            elif lead_params.get('pbc_x', False):  # wrap if periodic
                lead[lat(x,y,0), lat(0,y,0)] = lambda s1,s2,p=lead_params: hopping_x(s1,s2,p)

            # +y
            if y + 1 < L:
                lead[lat(x,y,0), lat(x,y+1,0)] = lambda s1,s2,p=lead_params: hopping_y(s1,s2,p)
            elif lead_params.get('pbc_y', False):
                lead[lat(x,y,0), lat(x,0,0)] = lambda s1,s2,p=lead_params: hopping_y(s1,s2,p)

            # Hopping to next z layer handled by symmetry
            lead[lat(x,y,0), lat(x,y,1)] = lambda s1,s2,p=lead_params: hopping_z(s1,s2,p)
    lead.eradicate_dangling()
    return lead

def syst_with_leads(L,p=params):

    syst = crystalline_3D(L,params=p)
    lead_plus = lead_in_z(L,params=p)

    lead_minus = lead_plus.reversed()

    syst.attach_lead(lead_plus)          # to layer k=0
    syst.attach_lead(lead_minus) 



    return syst.finalized(),lead_plus.finalized(),lead_minus.finalized()


# def make_system(p=params):

#     #We define   the cubic lattice with a=1
#     lat = kwant.lattice.general([a1,a2,a3],norbs=2*2)

#     #We define the symmetry
#     sym=kwant.TranslationalSymmetry(
#     lat.vec((1,0,0)), lat.vec((0,1,0)), lat.vec((0,0,1))) 

#     p = {**params, **p} 

#     MJ,lambdaJ,BJ,break_PH = p['MJ'], p['lambdaJ'],p['BJ'],p['break_PH']
#     print('BJ:',BJ,'MJ:',MJ,'break_PH:',break_PH)

#     model=kwant.Builder(sym)
    
#     W=p['W']
#     np.random.seed(int(p['seed_W']))
#     disorder = np.random.uniform(-W/2, W/2)

#     #We define the onsite energy:
#     model[lat(0,0,0)]=break_PH*kron(tau_0,sigma_0)+MJ*kron(tau_z,sigma  _0)+kron(tau_0,(BJ/sqrt(3))*(sigma_x + sigma_y + sigma_z))+ disorder*np.eye(4)

#     #We define the hoppings:
#     model[lat(0,0,0),lat(1,0,0)]=kron(tau_z,sigma_0)*(1/2) - 1j*lambdaJ*kron(tau_x,sigma_x)*(1/2)
#     model[lat(0,0,0),lat(0,1,0)]=kron(tau_z,sigma_0)*(1/2) - 1j*lambdaJ*kron(tau_x,sigma_y)*(1/2)
#     model[lat(0,0,0),lat(0,0,1)]=kron(tau_z,sigma_0)*(1/2) - 1j*lambdaJ*kron(tau_x,sigma_z)*(1/2)
#     return model


# def kwant_cubic_model(p=params):
#     #We define  the cubic lattice with a=1
#     lat = kwant.lattice.general([a1,a2,a3],norbs=2*2)

#     #We define the symmetry
#     sym=kwant.TranslationalSymmetry(
#         lat.vec((1,0,0)), lat.vec((0,1,0)), lat.vec((0,0,1))) 

#     p = {**params, **p} 
#     l1=l2=l3= p['L']
#     print('l:',l1)

#     def cuboid_shape(site):
#         x, y, z = site.pos
#         return 0 <= x <=l1 and 0 <= y <= l2 and 0 <= z <= l3
            

#     syst = kwant.Builder()

#     syst.fill(make_system(p),cuboid_shape, lat(0,0,0))

#     return syst

