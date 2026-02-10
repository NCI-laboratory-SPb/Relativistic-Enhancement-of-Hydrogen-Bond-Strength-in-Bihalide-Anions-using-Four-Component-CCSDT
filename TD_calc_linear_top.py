import numpy as np
from scipy.constants import R, h, c, k, N_A, u  

def calculate_thermo_properties(
    vibrational_frequencies,  
    B,                        
    sigma,                    
    mass_amu,                 
    pressure,                 
    temperature               
):
    
    R_J = R  
    hc = h * c * 1e2  
    k_J = k  
    mass_kg = mass_amu * u  

   
    pressure_Pa = pressure * 1e5

    
    ZPE_cm = 0.5 * np.sum(vibrational_frequencies)
    ZPE_J = ZPE_cm * hc * N_A
    ZPE_kJ = ZPE_J / 1000  
    ZPE_kcal = ZPE_kJ * 0.239
    
    q_rot = k_J * temperature / (sigma * B * hc)

   
    theta_vib = vibrational_frequencies * hc / k_J
    q_vib = np.prod(1 / (1 - np.exp(-theta_vib / temperature)))

    
    q_trans = (2 * np.pi * mass_kg * k_J * temperature / h**2)**(3/2) * k_J * temperature / pressure_Pa

    
    H_vib = R_J * temperature * np.sum((theta_vib / temperature)/ (np.exp(theta_vib / temperature) - 1))
    H_rot = R_J * temperature
    H_trans = 2.5 * R_J * temperature
    H_total = H_vib + H_rot + H_trans + ZPE_J
    H_kJ = H_total / 1000  
    H_kcal = H_kJ * 0.239
    
    S_trans = R_J * (np.log(q_trans) + 2.5)
    S_rot = R_J * (np.log(q_rot) + 1)
    S_vib = R_J * np.sum(((theta_vib / temperature) / (np.exp(theta_vib / temperature) - 1)) - np.log(1 - np.exp(-theta_vib / temperature)))
    S_total = S_trans + S_rot + S_vib  
   
    
    G_total = H_total - temperature * S_total
    G_kJ = G_total / 1000  
    G_kcal = G_kJ * 0.239
    return {
        "ZPE (kJ/mol)": ZPE_kJ,
        "ZPE (kcal/mol)": ZPE_kcal,
        "total H (kJ/mol)": H_kJ,
        "total H (kcal/mol)": H_kcal,
        "total G (kJ/mol)": G_kJ,
        "total G (kcal/mol)": G_kcal,
        "total S (J/(mol·K))": S_total,
        "translational part of S (J/(mol·K))": S_trans,
        "rotational part of S (J/(mol·K))": S_rot,
        "vibrational part of S (J/(mol·K))": S_vib,
    }

# Molecular constants for linear top
vibrational_frequencies = np.array([2579.2746])  # frequencies, 1/cm
B = 8.431258 # rotational constant, 1/cm
sigma = 1 # order of rotational subgroup of point group
mass_amu = 80.9123 # total molecular mass, a.u.
pressure = 1.0  # pressure, bar
temperature = 298.15  # temperature, K

results = calculate_thermo_properties(vibrational_frequencies, B, sigma, mass_amu, pressure, temperature)
for key, value in results.items():
    print(f"{key}: {value:.4f}")
