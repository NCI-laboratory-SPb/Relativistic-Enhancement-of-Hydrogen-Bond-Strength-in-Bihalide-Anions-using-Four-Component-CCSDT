import os
import numpy as np
from argparse import ArgumentParser
from scipy.linalg import eigh_tridiagonal

bohr_to_Ang = 0.529177249
hartree_to_kcal_per_mol = 627.5099746309728
hartree_to_joule = 4.3597482 * 10 ** -18
planck_constant = 6.625 * 10 ** -34
avogadro_number = 6.02214076 * 10 ** 23
speed_of_light = 299792458
mass_proton = 1836.152673425606
mass_proton_SI = 1.6726219236951 * 10 ** -27

def SEsolver_Rovib(coordinate, potential, mass, J):

    N = len(potential)
    L = max(coordinate) - min(coordinate)
    y = (coordinate - min(coordinate)) / L + 10 ** -20
    dy = abs(y[0] - y[1])
    m = mass_proton * mass

    d = 1 / dy ** 2 + m * ((L / bohr_to_Ang) ** 2) * (potential + J * (J + 1) / (2 * m * (coordinate / bohr_to_Ang) ** 2))
    e = -1 / (2 * dy ** 2) * np.ones(len(d) - 1)
    w, v = eigh_tridiagonal(d, e)

    return hartree_to_joule * w / (m * (L / bohr_to_Ang) ** 2), v

def main():
    parser = ArgumentParser()
    parser.add_argument("-f", dest="F", default=False, type=str,
                        help="Directory to the file with coordinate values and corresponding potential")
    parser.add_argument("-d", dest="polynom_degree", default=10, type=int, metavar="POLYNOM_DEGREE",
                        help="Degree of a polynom for potential approximation (default = 10)")
    parser.add_argument("--min", dest="min", default=False, type=float,
                        help="Minimal value for coordinate axis")
    parser.add_argument("--max", dest="max", default=False, type=float,
                        help="Maximal value for coordinate axis")
    parser.add_argument("-b", dest="bins", default=200, type=int, metavar="BINS",
                        help="Number of bins for coordinate axis discretization (default = 200)")
    parser.add_argument("-m", dest="mass", default=1, type=float, metavar="MASS",
                        help="Reduced mass in proton mass units (default = 1)")
    parser.add_argument("-j", dest="J", default=0, type=int, metavar="J",
                        help="solving 1D-SE with rotational coupling for linear rotator")

    (options, args) = parser.parse_known_args()

    file = options.F
    data = np.loadtxt(file)
    pot = data[:, 1] - np.min(data[:, 1])
    c = np.polyfit(data[:, 0], pot, options.polynom_degree)
    potential_as_polynom = np.poly1d(c)

    if options.min == False:
        coord_min = np.min(data[:, 0])
    else:
        coord_min = options.min

    if options.max == False:
        coord_max = np.max(data[:, 0])
    else:
        coord_max = options.max

    coordinate = np.linspace(coord_min, coord_max, options.bins)
    potential = potential_as_polynom(coordinate)
    energies, vectors = SEsolver_Rovib(coordinate, potential, options.mass, options.J)

    de10 = energies[1] - energies[0]
    de21 = energies[2] - energies[1]
    vib_fundamental = de10 / planck_constant / speed_of_light
    vib_oberton_1 = de21 / planck_constant / speed_of_light

    mean_coordinate = np.sum(vectors.T[0] ** 2 * coordinate)
    maxprob_coordinate = coordinate[np.argmax(vectors.T[0] ** 2)]
    rotational_constant = np.sum(vectors.T[0] ** 2 * (planck_constant / 2 / np.pi) ** 2
                                 / (2 * mass_proton_SI * options.mass
                                    * (coordinate * 10 ** -10) ** 2))

    rotational_constant = rotational_constant / planck_constant / speed_of_light

    print('Frequency of fundamental transition: ',
          np.round(vib_fundamental / 100, 4), ' cm-1')
    print('Frequency of first oberton: ',
          np.round((vib_oberton_1 + vib_fundamental) / 100, 4), ' cm-1')
    print('Anharmonicity index: ',
          np.round(vib_oberton_1 / vib_fundamental, 4))
    print('Mean coordinate value: ',
          np.round(mean_coordinate,4), ' Å')
    print('Most probable coordinate value: ',
          np.round(maxprob_coordinate, 4), ' Å')
    print('Rotational constant of one-dimensional oscillator: ',
          np.round(rotational_constant / 100, 4), ' cm-1')

if __name__ == "__main__":
    main()