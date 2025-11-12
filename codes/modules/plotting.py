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