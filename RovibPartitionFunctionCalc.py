#!/usr/bin/env python3
"""
RovibPartitionFunctionCalc.py by Daniil A. Shitov

Script for calculating thermodynamic functions (ZPE, enthalpy, Gibbs free energy, entropy)
based on molecular constants. Supports three types of systems:
  - atomic (point group SO3) – only translational contribution;
  - linear molecules (D∞h, C∞v) – one rotational constant B;
  - nonlinear molecules (other point groups) – three rotational constants A, B, C.

The symmetry number (σ) is automatically determined from the point group according to the table.

Usage:
    RovibPartitionFunctionCalc.py -PG=point_group -M=mass -T=temperature -p=pressure [additional keys]

Required keys:
    -PG=...   Schönflies point group (e.g., Dinfh, C2v, Td, SO3)
    -M=...    molecular mass in atomic mass units (a.m.u.)
    -T=...    temperature in Kelvin
    -p=...    pressure in bar

Additional keys for linear molecules (PG = Dinfh or Cinfv):
    -B=...      rotational constant, cm⁻¹
    -freq=[...] list of vibrational frequencies (including degeneracy), cm⁻¹, e.g., [100.1,3000.4]

For nonlinear molecules (all other PG except SO3):
    -A=...      first rotational constant, cm⁻¹
    -B=...      second rotational constant, cm⁻¹
    -C=...      third rotational constant, cm⁻¹
    -freq=[...] list of vibrational frequencies (including degeneracy), cm⁻¹

For atomic systems (PG = SO3) no other keys are required (if given, they are ignored with a warning).

Results are printed in molar quantities (per mole) in units of kJ/mol, kcal/mol and J/(mol·K).
"""

import sys
import numpy as np
from scipy.constants import R, h, c, k, N_A, u

# ------------------------------------------------------------
# Dictionary for symmetry number σ based on point group
# ------------------------------------------------------------
sigma_map = {
    # σ = 1
    'C1': 1, 'Cs': 1, 'Ci': 1, 'Cinfv': 1,
    # σ = 2
    'C2': 2, 'C2v': 2, 'C2h': 2, 'Dinfh': 2,
    # σ = 3
    'C3': 3, 'C3v': 3, 'C3h': 3,
    # σ = 4
    'C4': 4, 'C4v': 4, 'C4h': 4,
    'D2': 4, 'D2h': 4, 'D2d': 4,
    # σ = 5
    'C5': 5, 'C5v': 5, 'C5h': 5,
    # σ = 6
    'C6': 6, 'C6v': 6, 'C6h': 6,
    'D3': 6, 'D3h': 6, 'D3d': 6,
    # σ = 8
    'D4': 8, 'D4h': 8, 'D4d': 8,
    # σ = 10
    'D5': 10, 'D5h': 10, 'D5d': 10,
    # σ = 12
    'D6': 12, 'D6h': 12, 'D6d': 12,
    'T': 12, 'Td': 12,
    # σ = 24
    'O': 24, 'Oh': 24,
    # for atomic group (no rotation)
    'SO3': 1,
}

# ------------------------------------------------------------
# Helper functions for argument parsing
# ------------------------------------------------------------
def parse_args():
    """Parse command line arguments in the form -key=value."""
    args = {}
    for arg in sys.argv[1:]:
        if arg.startswith('-'):
            parts = arg[1:].split('=', 1)
            key = parts[0]
            if len(parts) == 2:
                value = parts[1]
            else:
                print(f"Error: argument {arg} must be in the form -key=value")
                sys.exit(1)
            args[key] = value
        else:
            print(f"Warning: ignoring argument without hyphen: {arg}")
    return args


def parse_freq(freq_str):
    """
    Convert a string like "[100.1,3000.4,3000.4,3500.1]" into a numpy array of numbers.
    Brackets are optional.
    """
    s = freq_str.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    if not s:
        return np.array([])
    parts = s.split(',')
    freqs = []
    for p in parts:
        try:
            freqs.append(float(p.strip()))
        except ValueError:
            print(f"Error: could not convert '{p}' to a number")
            sys.exit(1)
    return np.array(freqs)


# ------------------------------------------------------------
# Main calculation function
# ------------------------------------------------------------
def main():
    args = parse_args()

    # Check required keys
    required = ['PG', 'M', 'T', 'p']
    for r in required:
        if r not in args:
            print(f"Error: required key -{r} not provided")
            sys.exit(1)

    pg = args['PG']
    mass_amu = float(args['M'])
    temperature = float(args['T'])
    pressure = float(args['p'])

    # Determine symmetry number
    if pg not in sigma_map:
        print(f"Warning: point group {pg} not found in table, using σ = 1")
        sigma = 1
    else:
        sigma = sigma_map[pg]

    # Molecular type
    if pg == 'SO3':
        mol_type = 'atom'
    elif pg in ['Dinfh', 'Cinfv']:
        mol_type = 'linear'
    else:
        mol_type = 'nonlinear'

    # Obtain additional parameters based on type
    if mol_type == 'atom':
        # Warn if extra keys are provided
        for ek in ['B', 'A', 'C', 'freq']:
            if ek in args:
                print(f"Warning: for atomic system key -{ek} is ignored")
        vibrational_frequencies = np.array([])
        A = B = C = None
    elif mol_type == 'linear':
        if 'B' not in args:
            print("Error: for linear molecule key -B is required")
            sys.exit(1)
        if 'freq' not in args:
            print("Error: for linear molecule key -freq is required")
            sys.exit(1)
        B = float(args['B'])
        vibrational_frequencies = parse_freq(args['freq'])
        A = C = None
    else:  # nonlinear
        for key in ['A', 'B', 'C', 'freq']:
            if key not in args:
                print(f"Error: for nonlinear molecule key -{key} is required")
                sys.exit(1)
        A = float(args['A'])
        B = float(args['B'])
        C = float(args['C'])
        vibrational_frequencies = parse_freq(args['freq'])

    # --------------------------------------------------------
    # Physical constants and unit conversions
    # --------------------------------------------------------
    R_J = R                     # J/(mol·K)
    hc = h * c * 1e2            # J·cm  (energy corresponding to 1 cm⁻¹)
    k_J = k                     # J/K
    mass_kg = mass_amu * u      # kg
    pressure_Pa = pressure * 1e5  # Pa

    # --------------------------------------------------------
    # Zero-point vibrational energy (ZPE)
    # --------------------------------------------------------
    if len(vibrational_frequencies) > 0:
        ZPE_cm = 0.5 * np.sum(vibrational_frequencies)
        ZPE_J = ZPE_cm * hc * N_A      # J/mol
    else:
        ZPE_J = 0.0
    ZPE_kJ = ZPE_J / 1000
    ZPE_kcal = ZPE_kJ * 0.239          # approximate conversion (1/4.184)

    # --------------------------------------------------------
    # Translational partition function and contributions
    # --------------------------------------------------------
    q_trans = (2 * np.pi * mass_kg * k_J * temperature / h**2)**(3/2) * k_J * temperature / pressure_Pa
    H_trans = 2.5 * R_J * temperature
    S_trans = R_J * (np.log(q_trans) + 2.5)

    # --------------------------------------------------------
    # Rotational contributions (if applicable)
    # --------------------------------------------------------
    H_rot = 0.0
    S_rot = 0.0
    if mol_type == 'linear' and B > 0:
        q_rot = k_J * temperature / (sigma * B * hc)
        H_rot = R_J * temperature
        S_rot = R_J * (np.log(q_rot) + 1)
    elif mol_type == 'nonlinear' and A > 0 and B > 0 and C > 0:
        q_rot = np.sqrt(np.pi) * (k_J * temperature)**1.5 / (sigma * np.sqrt(A * hc * B * hc * C * hc))
        H_rot = 1.5 * R_J * temperature
        S_rot = R_J * (np.log(q_rot) + 1.5)

    # --------------------------------------------------------
    # Vibrational contributions
    # --------------------------------------------------------
    H_vib = 0.0
    S_vib = 0.0
    if len(vibrational_frequencies) > 0:
        theta_vib = vibrational_frequencies * hc / k_J   # characteristic temperature (K)
        # avoid overflow – for large temperatures exp(-theta/T) is small
        exp_arg = -theta_vib / temperature
        q_vib = np.prod(1.0 / (1.0 - np.exp(exp_arg)))
        # energy and entropy
        x = theta_vib / temperature
        exp_x = np.exp(x)
        H_vib = R_J * temperature * np.sum(x / (exp_x - 1.0))
        S_vib = R_J * np.sum(x / (exp_x - 1.0) - np.log(1.0 - np.exp(-x)))

    # --------------------------------------------------------
    # Total thermodynamic functions
    # --------------------------------------------------------
    H_total = H_trans + H_rot + H_vib + ZPE_J
    S_total = S_trans + S_rot + S_vib
    G_total = H_total - temperature * S_total

    H_kJ = H_total / 1000
    H_kcal = H_kJ * 0.239
    G_kJ = G_total / 1000
    G_kcal = G_kJ * 0.239

    # --------------------------------------------------------
    # Output results
    # --------------------------------------------------------
    print(f"ZPE (kJ/mol): {ZPE_kJ:.4f}")
    print(f"ZPE (kcal/mol): {ZPE_kcal:.4f}")
    print(f"total H (kJ/mol): {H_kJ:.4f}")
    print(f"total H (kcal/mol): {H_kcal:.4f}")
    print(f"total G (kJ/mol): {G_kJ:.4f}")
    print(f"total G (kcal/mol): {G_kcal:.4f}")
    print(f"total S (J/(mol·K)): {S_total:.4f}")
    print(f"translational part of S (J/(mol·K)): {S_trans:.4f}")
    print(f"rotational part of S (J/(mol·K)): {S_rot:.4f}")
    print(f"vibrational part of S (J/(mol·K)): {S_vib:.4f}")


if __name__ == "__main__":
    main()
