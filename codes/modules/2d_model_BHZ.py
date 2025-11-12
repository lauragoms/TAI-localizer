import numpy as np
import kwant
params = dict(Lx = 5,
         Ly = 5,
         m = 4.,
         B = 1, 
         A = 1)


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

def system_2D_BHZ(params=params):
    ''' l i s m'''

    lat = kwant.lattice.square(norbs = 4)

    print('m/2B:',params['m'])

    def onsite(site):
        return (params['m'] - 4*params['B'])* np.kron(sigma_z,sigma_0) + 0.1*np.kron(sigma_0,sigma_0)

    def hop_x(site0, site1):
        return (params['B']*np.kron(sigma_z,sigma_0) + params['A'] * np.kron(sigma_x,sigma_z)/2j)

    def	hop_y(site0, site1):
        return (params['B']*np.kron(sigma_z,sigma_0) - params['A'] * np.kron(sigma_y,sigma_0)/2j)

    def hop_pbc_y(site0, site1):
        return params['pbc']*hop_y(site0,site1)
    def hop_pbc_x(site0, site1):
        return params['pbc']*hop_x(site0,site1)
                                     
    sys = kwant.Builder()

    sys[(lat(x, y) for x in range(-params['Lx'],params['Lx']) for y in range(-params['Ly'],params['Ly']))] = onsite        
    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_x
    sys[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_y

    return sys 

def TR_op(len_sites):
    return np.kron(np.kron(np.identity(len_sites*len_sites),sigma_0),sigma_y)