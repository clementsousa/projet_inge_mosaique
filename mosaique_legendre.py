import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math


def multiplication_entier_gauss(a, b):
    return (a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0])


def symbole_legendre(x1, x2):
    '''Prend en paramètre x1, x2 qui sont les indices de parcours des deux premières boucles,
    qui permettent de déterminer si le couple qui appelle cette fonction est soit divisble par p,
    soit un résidu quadratique modulo p'''
    if (x1 == x2 == 0):
        # Divisble par p
        return 0
    else:
        # Résidu quadratique modulo p
        return 1


def creer_mosaique(p, m, n):
    # On initialise toutes les cases à -1 (ni multiple, ni résidu quadratique, donc -1 par définition du symbole de Legendre)
    M = np.ones((m+1, n+1))*-1

    p1, p2 = p[0], p[1]

    normeP = p1**2 + p2**2

    q = math.floor(math.sqrt(normeP/2))

    imax = math.floor(math.sqrt(2*normeP)*m/normeP)
    jmax = math.floor(math.sqrt(2*normeP)*n/normeP)

    # On parcourt tous les couples (x1, x2) qui sont des résidus quadratiques mod p ou divisibles par p
    for x1 in range(-q, q+1):
        for x2 in range(0, q+1):
            (a1, a2) = multiplication_entier_gauss((x1, x2), (x1, x2))

            r = (p1*(a1) + p2*(a2)) % normeP
            s = (p1*(a2) - p2*(a1 - m)) % normeP

            u0 = (p1*r - p2*(s - p2*m)) // normeP
            v0 = (p2*r + p1*(s - p2*m)) // normeP
            for k1 in range(0, imax+1):
                for k2 in range(0, jmax+1):
                    (u, v) = multiplication_entier_gauss((k1, k2), p)
                    u = int(u + u0)
                    v = int(v + v0)
                    # Le couple (u,v) est au moins un résidu quadratique mod p, au plus divisible par p
                    if (0 <= u <= m and 0 <= v <= n):
                        # Si le couple se situe dans la mosaïque
                        if(M[u, v] != 0): 
                            # Cette conditon évite qu'une case bleue (résidu quadratique) remplace une case rouge (multiple)
                            # Car un mutliple est un résidu quadratique particulier
                            M[u, v] = symbole_legendre(x1, x2)
    return M

# Paramètres à modifier pour avoir des résultats différents
p = (3, 8)
largeur = 67
hauteur = 67


resultat = creer_mosaique(p, largeur, hauteur)

couleur = ['white', 'red', 'darkblue']

plt.imshow(resultat.T, cmap=ListedColormap(couleur))
plt.gca().invert_yaxis()

plt.xticks(np.arange(-0.5, largeur, 1), minor=True)
plt.yticks(np.arange(-0.5, hauteur, 1), minor=True)

plt.grid(True, which='minor', color='grey', linestyle='-', linewidth=0.5)

plt.title(f"p = {p}")
plt.show()
