4cDCCCSDQTAIM.py
This script performs a four-component electron density calculation on a three-dimensional grid of points. The density is derived from a CCSD wave function generated with the Dirac-Coulomb and Lévy-Leblond Hamiltonians, incorporating a Gaussian finite-nucleus model.

Usage:

bash
4cDCCCSDQTAIM.py molecule.xyz [-DC | -LL] -dyall.BASIS
where molecule.xyz is one of: FHF.xyz, ClHCl.xyz, BrHBr.xyz, IHI.xyz
BASIS can be any of the Dyall basis sets:

v2z, cv2z, ae2z, av2z, acv2z, aae2z

v3z, cv3z, ae3z, av3z, acv3z, aae3z

v4z, cv4z, ae4z, av4z, acv4z, aae4z

Required keys:

-DC – Perform four-component calculations with Dirac-Coulomb Hamiltonian

-LL – Perform four-component calculations with Lévy-Leblond Hamiltonian

-dyall.BASIS – Expansion of the large and small components of one-electron bispinors in terms of a chosen uncontracted basis set of the Dyall family

XHXgennucconf.py
A code for generating nuclear configurations representing proton motion along normal modes in hydrogen bonds, under the constraint of a fixed center of mass.

Usage:

bash
XHXgennucconf.py [-FHF | -ClHCl | -BrHBr | -IHI] [-Q2 | -Q3]
Required keys:

-FHF – Generation of nuclear configurations for bifluoride anion

-ClHCl – Generation of nuclear configurations for bichloride anion

-BrHBr – Generation of nuclear configurations for bibromide anion

-IHI – Generation of nuclear configurations for biiodide anion

-Q2 – Generating nuclear configurations that represent proton motion perpendicular to the internuclear axis while keeping the center of mass fixed

-Q3 – Generating nuclear configurations that represent proton motion along the internuclear axis while keeping the center of mass fixed

RovibPartitionFunctionCalc.py
This program computes the translational, vibrational, and rotational components of thermodynamic functions using molecular constants (including the nuclear configuration point group, total molecular mass, vibrational frequencies, and rotational constants) together with external parameters (temperature and pressure). It supports atomic systems, linear tops, and asymmetric (nonlinear) tops.

Usage:

bash
RovibPartitionFunctionCalc.py -PG=point_group -M=mass -T=temperature -p=pressure [additional keys]
Required keys:

-PG=... – Schönflies point group (e.g., Dinfh, C2v, Td, SO3)

-M=... – molecular mass in atomic mass units (a.m.u.)

-T=... – temperature in Kelvin

-p=... – pressure in bar

Additional keys for linear molecules (PG = Dinfh or Cinfv):

-B=... – rotational constant, cm⁻¹

-freq=[...] – list of vibrational frequencies (including degeneracy), cm⁻¹

For nonlinear molecules (all other PG except SO3):

-A=... – first rotational constant, cm⁻¹

-B=... – second rotational constant, cm⁻¹

-C=... – third rotational constant, cm⁻¹

-freq=[...] – list of vibrational frequencies (including degeneracy), cm⁻¹

1DSESolver.py
A Python script for numerical solution of the one-dimensional time-independent Schrödinger equation with optional rovibrational coupling for linear rotators. No installation required. Simply download the script and run it with Python. Only NumPy and SciPy are required.

Usage:

bash
python 1DSESolver.py -f potential.dat [options]
Keys:

-f FILE – Path to input file with coordinate (Å) and potential (Hartree) [REQUIRED]

-d POLYNOM_DEGREE – Degree of polynomial for potential fitting (default: 10)

--min MIN_COORD – Minimum coordinate value for grid (default: min from input file)

--max MAX_COORD – Maximum coordinate value for grid (default: max from input file)

-b BINS – Number of grid points (default: 200)

-m MASS – Reduced mass in proton mass units (default: 1)

-j J – Rotational quantum number (default: 0)
