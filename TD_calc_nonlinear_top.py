import numpy as np
from scipy.constants import R, h, c, k, N_A, u  

def calculate_thermo_properties(
    vibrational_frequencies,  
    A, 
    B,
    C,                       
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
    
    q_rot = (np.pi)**(1/2)*(k_J * temperature)**(3/2)/(sigma * (A * hc * B * hc * C * hc)**(1/2))

   
    theta_vib = vibrational_frequencies * hc / k_J
    q_vib = np.prod(1 / (1 - np.exp(-theta_vib / temperature)))

    
    q_trans = (2 * np.pi * mass_kg * k_J * temperature / h**2)**(3/2) * k_J * temperature / pressure_Pa

    
    H_vib = R_J * temperature * np.sum((theta_vib / temperature)/ (np.exp(theta_vib / temperature) - 1))
    H_rot = 1.5 * R_J * temperature
    H_trans = 2.5 * R_J * temperature
    H_total = H_vib + H_rot + H_trans + ZPE_J
    H_kJ = H_total / 1000  
    H_kcal = H_kJ * 0.239
    
    S_trans = R_J * (np.log(q_trans) + 2.5)
    S_rot = R_J * (np.log(q_rot) + 1.5)
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

# Molecular constants 
vibrational_frequencies = np.array([2916.5, 1533.6, 1533.6, 3018.9, 3018.9, 3018.9, 1305.9, 1305.9, 1305.9])  # frequencies with accounting of degeneracy, 1/cm
A = 5.241  # 1st rotational constant (relative to the principal axis), 1/cm
B = 5.241  # 2nd rotational constant (relative to the principal axis), 1/cm
C = 5.241  # 3d rotational constant (relative to the principal axis), 1/cm
sigma = 12 # order of rotational subgroup
mass_amu = 16  # total molecular mass, a.u.
pressure = 1.0  # pressure, bar
temperature = 2000  # temperature, K

results = calculate_thermo_properties(vibrational_frequencies, A, B, C, sigma, mass_amu, pressure, temperature)
for key, value in results.items():
    print(f"{key}: {value:.4f}")
