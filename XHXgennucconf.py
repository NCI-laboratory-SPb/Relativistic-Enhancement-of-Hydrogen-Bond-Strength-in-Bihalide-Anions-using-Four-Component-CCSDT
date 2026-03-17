#!/usr/bin/env python3
"""
XHXgennucconf.py by Daniil A. Shitov

Generate nuclear configurations for bihalide anions [X–H–X]⁻ (X = F, Cl, Br, I)
with the center of mass fixed at the origin.
For a given anion and a displacement mode (Q2 or Q3), produces 20 xyz files
where the hydrogen nucleus is moved along the direction perpendicular to the
internuclear axis (Q2) or along the internuclear axis (Q3)
in steps of 0.01 Å from 0.01 to 0.20 Å. The halogen nuclei are shifted
accordingly to keep the center of mass at (0,0,0).

Usage:
    XHXgennucconf.py [-FHF | -ClHCl | -BrHBr | -IHI] [-Q2 | -Q3]
"""

import os
import sys
import argparse

# Nuclear masses of the most stable isotopes (in atomic mass units)
MASSES = {
    'H': 1.00782503223,   # hydrogen-1
    'F': 18.998403162,    # fluorine-19
    'Cl': 34.96885268,    # chlorine-35
    'Br': 78.9183376,     # bromine-79
    'I': 126.904473       # iodine-127
}

# Equilibrium positions (in Å) for each anion, assuming hydrogen at origin,
# halogens symmetric on the internuclear axis (z‑axis).
# Format: (halogen_symbol, internuclear_distance_halogen_from_origin)
EQUILIBRIUM = {
    'FHF':   ('F', 1.144262),
    'ClHCl': ('Cl', 1.569450),
    'BrHBr': ('Br', 1.714670),
    'IHI':   ('I', 1.905760)
}

def write_xyz(filename, nuclei, comment):
    """Write an xyz file with given nuclei (list of (symbol, x, y, z))."""
    with open(filename, 'w') as f:
        f.write(f"{len(nuclei)}\n")
        f.write(f"{comment}\n")
        for sym, x, y, z in nuclei:
            f.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate nuclear configurations for [X–H–X]⁻ anions')
    group_anion = parser.add_mutually_exclusive_group(required=True)
    group_anion.add_argument('-FHF', action='store_true', help='Fluorine anion FHF⁻')
    group_anion.add_argument('-ClHCl', action='store_true', help='Chlorine anion ClHCl⁻')
    group_anion.add_argument('-BrHBr', action='store_true', help='Bromine anion BrHBr⁻')
    group_anion.add_argument('-IHI', action='store_true', help='Iodine anion IHI⁻')

    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument('-Q2', action='store_true', help='Generate Q2 configurations (displacement perpendicular to the internuclear axis)')
    group_mode.add_argument('-Q3', action='store_true', help='Generate Q3 configurations (displacement along the internuclear axis)')

    args = parser.parse_args()

    # Determine anion
    if args.FHF:
        anion = 'FHF'
    elif args.ClHCl:
        anion = 'ClHCl'
    elif args.BrHBr:
        anion = 'BrHBr'
    else:
        anion = 'IHI'

    # Get data
    hal_sym, d = EQUILIBRIUM[anion]
    mH = MASSES['H']
    mX = MASSES[hal_sym]

    # Mode
    mode = 'Q2' if args.Q2 else 'Q3'
    axis_description = "perpendicular to the internuclear axis" if args.Q2 else "along the internuclear axis"

    # Generate 20 values from 0.01 to 0.20 step 0.01
    steps = [round(0.01 * i, 2) for i in range(1, 21)]

    for i, delta in enumerate(steps, start=1):
        if mode == 'Q2':
            # Displace hydrogen along x (perpendicular direction)
            xH = delta
            # Halogens move opposite along x to keep center of mass
            xX = - (mH * xH) / (2.0 * mX)
            # y and z unchanged
            nuclei = [
                (hal_sym, xX, 0.0,  d),
                (hal_sym, xX, 0.0, -d),
                ('H',     xH, 0.0, 0.0)
            ]
            comment = f"{anion} Q2 configuration, displacement = {delta:.2f} Å perpendicular to internuclear axis"
        else:  # Q3
            # Displace hydrogen along z (internuclear axis)
            zH = delta
            # Halogens move opposite along z
            dz = (mH * zH) / (2.0 * mX)  # positive shift to be subtracted
            # Their new z coordinates
            zX1 =  d - dz
            zX2 = -d - dz
            nuclei = [
                (hal_sym, 0.0, 0.0, zX1),
                (hal_sym, 0.0, 0.0, zX2),
                ('H',     0.0, 0.0, zH)
            ]
            comment = f"{anion} Q3 configuration, displacement = {delta:.2f} Å along internuclear axis"

        filename = f"{anion}_{mode}_{i:02d}.xyz"
        write_xyz(filename, nuclei, comment)
        print(f"Generated {filename}")

if __name__ == "__main__":
    main()