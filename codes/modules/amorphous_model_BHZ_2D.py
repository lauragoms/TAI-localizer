import numpy as np
import kwant
import copy
from scipy import spatial

params = dict(norbs = 4,
                Lx = 5,
                Ly = 5,
                Delta = 4.,
                B = 1, 
                A = 1,
                mu = 0,
                mu_leads = 0,
                dis_onsite = 0,
                seed_onsite = 0,
                dis_hadamard = 0,
                seed_hadamard=0,
                R = 1.1,
                name = 'a',
                dis_amorph = 0,
                seed_amorph = 0)

'''helper functions'''

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

def polar_coords(x0, y0):
    ''' Determines spherical coordinates of a point with
    respect to 0.    

    Parameters
    -----------------
    x0, y0 : floats,
        2D position of the point 
    
    Returns
    ----------------
    rho,  phi : floats,
        spherical coordinates'''
    
    rho = np.sqrt((x0)**2+(y0)**2)

    phi=np.sign(y0)*np.arccos(x0/np.sqrt(x0**2+y0**2))


    return rho, phi

'''hoppings'''

def onsite(p,rng_W):
    W = p['dis_onsite']
    disorder = rng_W.uniform(-W/2, W/2)
    return (p['Delta'] - 4*p['B'])* np.kron(sigma_z,sigma_0) + disorder * np.eye(p['norbs']) + p['mu']*np.kron(sigma_0,sigma_0) 


def amorph_hopping(site0,site1, vec, p, rng_hdmd):

    spin = sigma_z
    if p['dis_hadamard']!=0 and rng_hdmd.choice([0, 1], p=[1 - p['dis_hadamard']/100, p['dis_hadamard']/100])==1:
        spin = sigma_x
    
    rho,phi = polar_coords(vec[0],vec[1])
    hop_strength = p['B']*np.kron(sigma_z,sigma_0)
    hop_x_SOC =  np.cos(phi)*(p['A']/(2j) * np.kron(sigma_x,spin))
    hop_y_SOC =  - np.sin(phi)*p['A']/(2j) * np.kron(sigma_y,sigma_0)

    return hop_strength + hop_x_SOC + hop_y_SOC

'''amorphous system'''

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
    
     
def Displacement_2D(sites,seed,sigma):
    
    '''Given a crystalline system, the sites are displaced by a Gaussian distribution with standard deviation sigma

    Parameters
    -----------------
    sites : fsyst.sites
        List of sites from kwant builder
    sigma : float
        Standar deviation for the Gaussian displacement of sites

    Returns
    -----------------
    amorph_a,amorph_b,Sites
    '''

    disp_sites = []
    np.random.seed(int(seed))
    print('seed σ:',seed)
    disp = np.random.normal(0,sigma, size = 2 * len(sites))

    print('sigma: ',sigma)
    for i in range(len(sites)):

        x, y = sites[i][0], sites[i][1]

        disp_x, disp_y= disp[2*i:2*i+2]
        x = x + disp_x
        y = y + disp_y
        disp_sites.append(np.array([x,y]))

    return disp_sites


def Bond_2D(lat,p=params):
    '''
        Returns the bonds of the lattice.

        Parameters
        ------------
        lat : list
            List of sites in the lattice
        p : dict
            Params.

        Returns
        -----------
        info_bond : list
            List of tags of the bounded sites.
    '''

    def distance(vec_1,vec_2):
        vec_distance = vec_1 - vec_2
        dist_size = np.sqrt(np.dot(vec_distance,vec_distance))
        return vec_distance,dist_size #returns both the vector and its length
    og_size = len(lat) #how many sites in OBC

    out = copy.deepcopy(lat)

    tree = spatial.KDTree(out) #This class provides an index into a set of k-dimensional points which can be used to rapidly look up the nearest neighbors of any point.
    # Theta function in the hoppings
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
                    vec_dist, dist = distance(np.array(out[item]),np.array(out[i])) 
                    info_bond.append([i,new_index,vec_dist])
                    info_bond2.append((i,new_index))
                    a.append(new_index)

    return info_bond

def amorph_BHZ(Ly,Lx,p=params,finalize=False):

    """
    Returns an amorphous 2D BHZ model system with given parameters.

    Parameters
    ----------
    sites : list
        List of position of the amorphous system.
    p : dict
        Dictionary containing the model parameters {A,B,Delta,dis_onsite,dis_hadamard}

    Returns
    -------
    syst : kwant.builder.Builder
    """
    sites = [(x,y) for x in range(-Lx,Lx) for y in range(-Ly,Ly)] 
    sites = Displacement_2D(sites,sigma=p['dis_amorph'],seed=p['seed_amorph'])
    lat = Amorphous(sites)
    bonds = Bond_2D(sites,p=p)
    syst = kwant.Builder()

    rng_W = np.random.default_rng(int(p['seed_onsite']))
    rng_hdmd = np.random.default_rng(int(p['seed_hadamard']))

    for i in range(len(sites)):
        syst[lat(i)] = onsite(p,rng_W)

    for i,j,vec in bonds:
        syst[lat(i),lat(j)] = amorph_hopping(lat(i),lat(j),vec,p,rng_hdmd)

    return syst.finalized() if finalize else syst