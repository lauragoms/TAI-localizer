
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
            L=6,
            W=0,
            break_PH=0.05,
            norbs=4,
            name='a',
            seed=0,
            seed_W=0,
            pbc_x=False,pbc_y=False,pbc_z=False, 
            mu_leads=0.7,
            mu=0)


    

def onsite_term(params=params):
    MJ, lambdaJ, BJ, break_PH, W = params['MJ'], params['lambdaJ'], params['BJ'], params['break_PH'], params['W']
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

    if params['W']!=0:
        print('Disordered system with strength',params['W'],'with seed',params['seed_W'])

    np.random.seed(int(params['seed_W']))
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

    lead_params['W'] = 0
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
#     model[lat(0,0,0)]=break_PH*kron(tau_0,sigma_0)+MJ*kron(tau_z,sigma_0)+kron(tau_0,(BJ/sqrt(3))*(sigma_x + sigma_y + sigma_z))+ disorder*np.eye(4)

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

'''Amorphous Lattice Definition'''

class Amorphous(kwant.builder.SiteFamily):
    '''Creates a lattice from the positions of sites.'''
    def __init__(self, coords):
        n = params['name']
        self.coords = coords
        super(Amorphous, self).__init__(str(n+n), str(n), params['norbs']) # The __init__ method is crucial in object-oriented programming in Python. It is a special method automatically called when an object is created from a class. This method allows us to initialize an object's attributes and perform any necessary setup or initialization tasks.
    def normalize_tag(self, tag):
        try: tag = int(tag[0])
        except: raise KeyError     
        if 0 <= tag < len(self.coords):
            return tag
        else: raise KeyError
    def pos(self, tag):
        return self.coords[tag]
    def family(self):
        n = params['name']
        return str(n)
    

def Bond_3D(lat,p=params):

    """function to return whom is bonded to whom
        Parameters:
        lat: all the sites in the lattice
        PBC: whether or not it has PBC
        r: the cut-off distance for bonding
        size: the OBC sample size
    """
    def distance(vec_1,vec_2):
        vec_distance = vec_1 - vec_2
        dist_size = np.sqrt(np.dot(vec_distance,vec_distance))
        return vec_distance,dist_size #returns both the vector and its length
    
    p = {**params, **p} 
    
    og_size = len(lat) #how many sites in OBC

    lx = p['L'] + 1
    ly = p['L'] + 1
    lz = p['L'] + 1
    #Regular PBC
    orientationsx = [0,-lx,lx]
    orientationsy = [0,-ly,ly]
    orientationsz = [0,-lz,lz]
    out = copy.deepcopy(lat)
    if p['pbc']:
        print('Periodic Boundary Conditions')
        for off in list(product(orientationsx,orientationsy,orientationsz))[1:]:
        #for off in orientations:
            off = np.array(off)
            out = np.concatenate((out,lat+off)) #out is a list with all the PBC copies of lat
    tree = spatial.KDTree(out)
    bonds = tree.query_ball_point(x = lat, r = p['R']) #find points in lat that are within distance r of out
    info_bond = list()
    info_bond2 = list()
    
    for i in range(len(bonds)):
        b = bonds[i]
        b.remove(i) #remove onsite
        a = list()
        for item in b:
            new_index = item
            if item >= og_size: #not in OBC
                new_index = item - int(item/og_size)*og_size #readjusts from a PBC copy to the index in the OBC
            if i < new_index:
                    vec_dist, dist = distance(out[i],out[item])
                    info_bond.append([i,new_index,vec_dist]) # if you have pbc, then the distance has to be extracted from here
                    # print(i,new_index,dist,out[i],out[item])
                    a.append(new_index)
        info_bond2.append(a)
    return info_bond,info_bond2

def cubic_lattice(p=params):
    """
    Creates a cubic lattice of points with lattice constant a and side length L.

    Returns:
    - np.ndarray: Array of shape (N, 3), where N is the number of lattice points.
    """
    p = {**params, **p} 
    L = p ['L'] + 1
    a = 1 #lattice constant
    num_points = int(np.floor(L / a))
    offset = (num_points - 1) * a / 2  # shift to center the cube at origin
    coords = np.arange(num_points) * a - offset
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    return points



def create_cubic_TI(p=params):
    p = {**params, **p} 

    Sites = cubic_lattice(p)
    bonds = Bond_3D(Sites,p=p)[0]

    params['name'] = 'a'
    lat = Amorphous(Sites)

    N = len(Sites)

    syst = kwant.Builder()

    MJ,lambdaJ,BJ,break_PH,W,seed = p['MJ'], p['lambdaJ'],p['BJ'],p['break_PH'],p['W'],p['seed']
    print('BJ:',BJ,'MJ:',MJ,'break_PH:',break_PH,'W:',W,'seed:',seed)

    np.random.seed(int(seed))
    disorder = np.random.uniform(-W/2, W/2, len(Sites))
    # Onsite energy
    for i in range(N):
        syst[lat(i)] = break_PH*kron(tau_0,sigma_0)+MJ*kron(tau_z,sigma_0)+kron(tau_0,(BJ/sqrt(3))*(sigma_x + sigma_y + sigma_z)) + disorder[i]*kron(tau_0,sigma_0)

    def hopping(site0,site1,vec):
        ''' site0 = target site
            site1 = origin site'''
        
        if vec[0] == 0 and vec[1] == 0 and vec[2] == 0:
            return np.zeros((p['norbs'],p['norbs']),dtype=complex)
        
        else:

            SIGMA = -(vec[0]*sigma_x + vec[1]*sigma_y + vec[2]*sigma_z)

            return kron(tau_z,sigma_0)*(1/2) - 1j*lambdaJ*kron(tau_x,SIGMA)*(1/2)
        
    # Hopping terms 
    for i,j,vec in bonds:
        # print(i,j,vec,hopping(lat(i),lat(j),vec))

        syst[lat(i),lat(j)] = hopping(lat(i),lat(j),vec)

    return syst



'''Bloch Hamiltonian'''

def Bloch_H(kx,ky,kz,p=params):
    """
    Constructs the Bloch Hamiltonian for a cubic lattice with a given k-vector.
    """
    p = {**params, **p} 

    MJ,lambdaJ,BJ,break_PH = p['MJ'], p['lambdaJ'],p['BJ'],p['break_PH']
    # print('BJ:',BJ,'MJ:',MJ,'break_PH:',break_PH)

    H1 = (MJ+ np.cos(kx) + np.cos(ky) + np.cos(kz) ) * kron(tau_z,sigma_0)
    H2 = lambdaJ*(np.sin(kx)*kron(tau_x,sigma_x) + np.sin(ky)*kron(tau_x,sigma_y) + np.sin(kz)*kron(tau_x,sigma_z))
    H3 = (BJ/sqrt(3))*(kron(tau_0,sigma_x) + kron(tau_0,sigma_y) + kron(tau_0,sigma_z))
    H_break_PH = break_PH*kron(tau_0,sigma_0)



    return H1 + H2 + H3 + H_break_PH