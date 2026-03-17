#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerical Solver for 1D Schrödinger Equation with Rovibrational Coupling
========================================================================

This script numerically solves the one-dimensional time-independent Schrödinger 
equation for a particle in a given potential. It includes an option to add 
rovibrational coupling term for linear rotators (J(J+1)/(2μr²) term).

The solver uses the finite difference method with tridiagonal matrix 
diagonalization (via scipy.linalg.eigh_tridiagonal) to find eigenvalues 
(energies) and eigenvectors (wavefunctions).

Features:
---------
- Reads potential energy curve from input file
- Optional polynomial fitting of the potential
- Adjustable coordinate range and discretization
- Rovibrational coupling (J quantum number)
- Computes vibrational frequencies and anharmonicity
- Calculates expectation values and rotational constants

Dependencies:
------------
- numpy
- scipy.linalg
- argparse

Author: [Your name]
Date: [Current date]
Version: 1.0
"""

import os
import numpy as np
from argparse import ArgumentParser
from scipy.linalg import eigh_tridiagonal

# Physical constants in various unit systems
bohr_to_Ang = 0.529177249  # Bohr radius to Angstroms
hartree_to_kcal_per_mol = 627.5099746309728  # Hartree to kcal/mol
hartree_to_joule = 4.3597482 * 10 ** -18  # Hartree to Joules
planck_constant = 6.625 * 10 ** -34  # Planck constant (J·s)
avogadro_number = 6.02214076 * 10 ** 23  # Avogadro's number
speed_of_light = 299792458  # Speed of light (m/s)
mass_proton = 1836.152673425606  # Proton mass in atomic units
mass_proton_SI = 1.6726219236951 * 10 ** -27  # Proton mass in kg


def SEsolver_Rovib(coordinate, potential, mass, J):
    """
    Solve the 1D Schrödinger equation with optional rovibrational coupling.
    
    Parameters:
    -----------
    coordinate : array_like
        Discretized coordinate grid (in Angstroms)
    potential : array_like
        Potential energy values at each coordinate point (in Hartree)
    mass : float
        Reduced mass in proton mass units
    J : int
        Rotational quantum number (0 for pure vibrational problem)
    
    Returns:
    --------
    energies : ndarray
        Energy eigenvalues (in Joules)
    wavefunctions : ndarray
        Corresponding wavefunctions (eigenvectors)
    """
    N = len(potential)
    L = max(coordinate) - min(coordinate)  # Grid length in Angstroms
    
    # Normalized coordinate transformation for numerical stability
    y = (coordinate - min(coordinate)) / L + 10 ** -20  # Add small offset to avoid division by zero
    dy = abs(y[0] - y[1])  # Grid spacing in normalized coordinates
    
    # Reduced mass in atomic units
    m = mass_proton * mass
    
    # Construct diagonal and off-diagonal elements of the Hamiltonian matrix
    # Diagonal elements: 1/dy² + m*(L/bohr_to_Ang)² * (V + rotational term)
    d = 1 / dy ** 2 + m * ((L / bohr_to_Ang) ** 2) * (
            potential + J * (J + 1) / (2 * m * (coordinate / bohr_to_Ang) ** 2)
    )
    
    # Off-diagonal elements: -1/(2dy²)
    e = -1 / (2 * dy ** 2) * np.ones(len(d) - 1)
    
    # Solve the tridiagonal eigenvalue problem
    w, v = eigh_tridiagonal(d, e)
    
    # Convert energies to Joules and scale properly
    energies = hartree_to_joule * w / (m * (L / bohr_to_Ang) ** 2)
    
    return energies, v


def main():
    """
    Main function: parse command line arguments, read input data,
    solve the Schrödinger equation, and output results.
    """
    parser = ArgumentParser(description='Numerical solver for 1D Schrödinger equation with rovibrational coupling')
    
    # Input/output parameters
    parser.add_argument("-f", dest="F", default=False, type=str,
                        help="Path to file with coordinate and potential values (two columns)")
    
    # Potential fitting parameters
    parser.add_argument("-d", dest="polynom_degree", default=10, type=int, metavar="POLYNOM_DEGREE",
                        help="Degree of polynomial for potential approximation (default = 10)")
    
    # Grid parameters
    parser.add_argument("--min", dest="min", default=False, type=float,
                        help="Minimum value for coordinate axis (in Angstroms)")
    parser.add_argument("--max", dest="max", default=False, type=float,
                        help="Maximum value for coordinate axis (in Angstroms)")
    parser.add_argument("-b", dest="bins", default=200, type=int, metavar="BINS",
                        help="Number of grid points for coordinate discretization (default = 200)")
    
    # Physical parameters
    parser.add_argument("-m", dest="mass", default=1, type=float, metavar="MASS",
                        help="Reduced mass in proton mass units (default = 1)")
    parser.add_argument("-j", dest="J", default=0, type=int, metavar="J",
                        help="Rotational quantum number for linear rotator coupling")

    (options, args) = parser.parse_known_args()

    # Read and process input data
    file = options.F
    data = np.loadtxt(file)
    
    # Shift potential to minimum (zero at equilibrium)
    pot = data[:, 1] - np.min(data[:, 1])
    
    # Fit potential to polynomial for smooth interpolation
    c = np.polyfit(data[:, 0], pot, options.polynom_degree)
    potential_as_polynom = np.poly1d(c)

    # Set coordinate grid boundaries
    if options.min == False:
        coord_min = np.min(data[:, 0])
    else:
        coord_min = options.min

    if options.max == False:
        coord_max = np.max(data[:, 0])
    else:
        coord_max = options.max

    # Create uniform coordinate grid
    coordinate = np.linspace(coord_min, coord_max, options.bins)
    
    # Evaluate potential on the grid
    potential = potential_as_polynom(coordinate)
    
    # Solve Schrödinger equation
    energies, vectors = SEsolver_Rovib(coordinate, potential, options.mass, options.J)

    # Calculate vibrational frequencies
    de10 = energies[1] - energies[0]  # Energy difference between v=1 and v=0
    de21 = energies[2] - energies[1]  # Energy difference between v=2 and v=1
    
    # Convert to wavenumbers (cm⁻¹)
    vib_fundamental = de10 / planck_constant / speed_of_light  # Fundamental frequency (v=0→1)
    vib_oberton_1 = de21 / planck_constant / speed_of_light    # First overtone (v=1→2)

    # Calculate expectation values for ground state
    mean_coordinate = np.sum(vectors.T[0] ** 2 * coordinate)  # ⟨r⟩
    maxprob_coordinate = coordinate[np.argmax(vectors.T[0] ** 2)]  # Most probable coordinate
    
    # Calculate rotational constant B = ⟨ħ²/(2μr²)⟩
    rotational_constant = np.sum(vectors.T[0] ** 2 * (planck_constant / 2 / np.pi) ** 2
                                 / (2 * mass_proton_SI * options.mass
                                    * (coordinate * 10 ** -10) ** 2))
    
    # Convert rotational constant to cm⁻¹
    rotational_constant = rotational_constant / planck_constant / speed_of_light

    # Output results
    print('Frequency of fundamental transition: ',
          np.round(vib_fundamental / 100, 4), ' cm-1')  # Divide by 100 to convert from m⁻¹ to cm⁻¹
    print('Frequency of first overtone: ',
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
