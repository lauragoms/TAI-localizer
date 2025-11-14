import scienceplots
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def map_kappaE_locgap(kappar,Er,locgap,title='Local gap map '):
    plt.style.use('science')

    plt.figure()

    cbar = plt.pcolormesh(kappar,Er,locgap,cmap=cm.plasma,norm='log',shading='gouraud')


    plt.title(title,fontsize=20)
    plt.xlabel('$\kappa$',fontsize=20)
    plt.ylabel('E',fontsize=20)
    plt.colorbar(cbar,label='Local Gap')

    plt.show()


def map_Delta_W_conductance(DeltaR,WR,G,title='Local gap map '):
    plt.style.use('science')

    plt.figure(figsize=(8,6))

    cbar = plt.pcolormesh(DeltaR,WR,G,cmap=cm.managua,shading='gouraud')


    plt.title(title,fontsize=20)
    plt.xlabel('$\Delta$',fontsize=20)
    plt.ylabel('W',fontsize=20)
    plt.colorbar(cbar,label='G')

    plt.show()


def map_Delta_W_localgap(DeltaR,WR,lg,title='Local gap map '):
    plt.style.use('science')

    plt.figure(figsize=(8,6))

    cbar = plt.pcolormesh(DeltaR,WR,lg,cmap=cm.cividis,shading='gouraud')


    plt.title(title,fontsize=20)
    plt.xlabel('$\Delta$',fontsize=20)
    plt.ylabel('W',fontsize=20)
    plt.colorbar(cbar,label='Local Gap')

    plt.show()