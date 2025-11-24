import numpy as np
import time
import pfaffian as pff



def pos_H(fsyst, coord=0):
    """ Calculate the position operator in the 'coord' direction of the Hamiltonian of fsyst. 
    """
    H, ton, fon = fsyst.hamiltonian_submatrix(return_norb=True)
    x = np.zeros(H.shape)
    ind = 0

    for i in range(len(fsyst.sites)):
        for j in range(ind, ind + ton[i]):

            x[j, j] =  fsyst.sites[i].pos[coord]
            # x0[j, j] = fsyst.sites[index_origin].pos[coord]
        ind += ton[i]

    return x

def get_center(fsyst_sites):
    sites = {}
    d = len(fsyst_sites[0].pos)

    for i in range(d):
        sites[i] = []
    
    for site in fsyst_sites:
        site = site.pos
        for coord in range(d):
            sites[coord].append(site[coord])
    if d==1:
        return (np.max(np.array(sites[0])) + np.min(np.array(sites[0])))/2
    if d==2:
        return (np.max(np.array(sites[0])) + np.min(np.array(sites[0])))/2,(np.max(np.array(sites[1])) + np.min(np.array(sites[1])))/2
    if d==3:
        return (np.max(np.array(sites[0])) + np.min(np.array(sites[0])))/2,(np.max(np.array(sites[1])) + np.min(np.array(sites[1])))/2,(np.max(np.array(sites[2])) + np.min(np.array(sites[2])))/2


def z2_spectral_localizer_AII2D(syst, W, E0, TR,
                            X0=np.array(['None']),num_reals=50,kappa=0.1):

    '''Computes the Z2 invariant given by the spectral localizer for a 2D AII system.

    Parameters
    ----------------
    syst : kwant object non-finalized
        Model of the system.
    W : float
        Onsite disorder.
    E0 : float
        Energy where the invariant is computed.
    TR : numpy.array
        Time Reversal operator of the system.
    X0  : numpy.array
        Position where the invariant is computed.
    num_reals : int
        Number of disorder realizations.
    p : dict
        Params for the spectral localizer (kappa,)

  
    Returns
    ----------------
    pfaffian_averaged : float
        Topological invariant as the sign of the pfaffian.
    list_pf_reals : list
        List of pfaffian for disorder realizations.
        
    '''

    fsyst = syst.finalized()

    ham = fsyst.hamiltonian_submatrix()

    if X0[0]=='None':
        X0 = get_center(fsyst.sites)
    # if flatten==True:
    #     ham = flattened_H(ham)


    x0,y0=X0

    X = pos_H(fsyst,coord=0)
    Y = pos_H(fsyst,coord=1)
    id = np.identity(np.shape(ham)[0])
    print('E0:',E0,'x0,y0:',x0,y0,'kappa:',kappa,'W:',W)

    D = (X-(x0+0.2)*id)+1j*(Y-(y0+0.2)*id)
    
    Q = (1/np.sqrt(2))*np.block([[id,TR],
                                 [-TR,id]])
    
    if W==0:
        num_reals = 1
    seed_range = np.arange(num_reals)

    pfaffian_realizations = 0
    list_pf_reals = []

    print('Averaging over realizations...')
    for ind,val in enumerate(seed_range):
        # update_progress((ind+1)/len(seed_range))

        seed = val
        np.random.seed(int(seed))
        # ANDERSON'S DISORDER
        if W!=0:
            disp = np.diag(np.random.uniform(low=-W/2,high=W/2,size=len(fsyst.sites)))
            AND_disorder = np.kron(disp,np.eye(4))

        if W==0:
            h=ham - E0*id 
        else:
            h = ham - E0*id  + AND_disorder 


        L = np.block([[h,kappa*np.conjugate(D)],
            [kappa*D,-h]])
        Hp = 1j*np.conjugate(Q)@L@Q  #lorings section 5.4 i*conj(Q).H.Q

        pfaff_sign = np.real(pff.pfaffian(Hp,sign_only=True))
        pfaffian_realizations = pfaffian_realizations + pfaff_sign
        list_pf_reals.append(pfaff_sign)
        print('Realization Pfaffian:',np.real(pfaff_sign))


    pfaffian_averaged = pfaffian_realizations/num_reals


    print('Pfaffian sign:',pfaffian_averaged)
    return pfaffian_averaged,list_pf_reals
