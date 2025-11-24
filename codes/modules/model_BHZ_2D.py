import numpy as np
import kwant


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
                seed_hadamard=0)


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

'''Crystalline system'''


def system_2D_BHZ(Lx,Ly,p=params,finalize=False):
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

    p = p.copy()
    lat = kwant.lattice.square(norbs = p['norbs'])

    hadamard = p['dis_hadamard']

    if p['dis_onsite']!=0:
        print('Onsite disorder with strength',p['dis_onsite'], 'and seed',p['seed_onsite'])

    if p['dis_hadamard']!=0:
        print('Spin disorder with probability',hadamard,'and seed',p['seed_hadamard'])

    rng_W = np.random.default_rng(int(p['seed_onsite']))
    rng_hdmd = np.random.default_rng(int(p['seed_hadamard']))

    def onsite(site):
        W = p['dis_onsite']
        disorder = rng_W.uniform(-W/2, W/2)
        return (p['Delta'] - 4*p['B'])* np.kron(sigma_z,sigma_0) + disorder * np.eye(p['norbs']) + p['mu']*np.kron(sigma_0,sigma_0) # peru's code is -1 because of ws, wp = -1
    

    def hop_x(site0, site1):
        spin = sigma_z
        # print(rng_hdmd.choice([0, 1], p=[1 - hadamard/100, hadamard/100])==1)
        
        if hadamard!=0 and rng_hdmd.choice([0, 1], p=[1 - hadamard/100, hadamard/100])==1:
            spin = sigma_x

        return (p['B']*np.kron(sigma_z,sigma_0) + p['A']/(2j) * np.kron(sigma_x,spin))

    def	hop_y(site0, site1):
        return (p['B']*np.kron(sigma_z,sigma_0) - p['A']/(2j) * np.kron(sigma_y,sigma_0))
                       
    sys = kwant.Builder()

    sys[(lat(x, y) for x in range(-Lx,Lx) for y in range(-Ly,Ly))] = onsite        
    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_x
    sys[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_y
    if finalize:
        sys = sys.finalized()
    return sys,lat


def lead_BHZ(Ly,lat,p=params):

    hadamard = p['hadamard']

    def onsite(site):
        return (p['Delta'] - 4*p['B'])* np.kron(sigma_z,sigma_0) + p['mu_leads']*np.kron(sigma_0,sigma_0) # peru's code is -1 because of ws, wp = -1
   
    def hop_x(site0, site1):
        if hadamard:
            return (p['B']*np.kron(sigma_z,sigma_0) + p['A']/(2j) * np.kron(sigma_x,sigma_x))
        else:
            return (p['B']*np.kron(sigma_z,sigma_0) + p['A']/(2j) * np.kron(sigma_x,sigma_z))

    def	hop_y(site0, site1):
        return (p['B']*np.kron(sigma_z,sigma_0) - p['A']/(2j) * np.kron(sigma_y,sigma_0))


    sym = kwant.TranslationalSymmetry(lat.vec((1,0)))
    lead = kwant.Builder(sym)
    
    lead[(lat(0, y) for y in range(-Ly,Ly))] = onsite        
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_x
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_y

    return lead

def BHZ_with_leads(Lx,Ly,p=params):
    
    syst, lat = system_2D_BHZ(Lx, Ly, p=p)
    lead_plus  = lead_BHZ(Ly, lat, p=p)

    lead_minus = lead_plus.reversed()

    syst.attach_lead(lead_plus)          # to layer k=0
    syst.attach_lead(lead_minus) 

    return syst.finalized(), lead_plus.finalized(), lead_minus.finalized()

def TR_op(len_sites):
    return np.kron(np.kron(np.identity(len_sites),sigma_0),sigma_y)



