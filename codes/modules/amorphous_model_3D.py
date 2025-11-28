
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

params=dict(pbc=False,
            R=1.01,
            MJ=2.3,
            lambdaJ=1,
            BJ=0.,
            break_PH=0.05,
            norbs=4,
            name='a',
            dis_onsite=0,
            seed_onsite=0,
            pbc_x=False,pbc_y=False,pbc_z=False, 
            mu_leads=0.7,
            mu=0)

tau_0 = sigma_0 = s_0 = np.array([[1, 0], [0, 1]])
tau_x = sigma_x = s_x = np.array([[0, 1], [1, 0]])
tau_y = sigma_y = s_y = np.array([[0, -1j], [1j, 0]])

tau_z = sigma_z = s_z = np.array([[1, 0], [0, -1]])

def onsite():
    break_PH  = break_PH*kron(tau_0,sigma_0)
    MJ*kron(tau_z,sigma_0)+kron(tau_0,(BJ/sqrt(3))*(sigma_x + sigma_y + sigma_z)) + disorder[i]*kron(tau_0,sigma_0)


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



def create_cubic_TI(Lx,Ly,Lz,p=params):
    p = {**params, **p} 

    # sites = cubic_lattice(p)
    sites = [(x,y,z) for x in range(-Lx,Lx) for y in range(-Ly,Ly) for z in range (-Lz,Lz)] 

    bonds = Bond_3D(sites,p=p)[0]

    params['name'] = 'a'
    lat = Amorphous(sites)

    N = len(sites)

    syst = kwant.Builder()

    MJ,lambdaJ,BJ,break_PH,W,seed_W = p['MJ'], p['lambdaJ'],p['BJ'],p['break_PH'],p['dis_onsite'],p['seed_onsite']
    print('BJ:',BJ,'MJ:',MJ,'break_PH:',break_PH,'W:',W,'seed:',seed_W)

    rng_W = np.random.default_rng(int(p['seed_onsite']))
    disorder = np.random.uniform(-W/2, W/2, len(sites))

    # Onsite energy
    for i in range(N):
        syst[lat(i)] = onsite(p,rng_W)

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