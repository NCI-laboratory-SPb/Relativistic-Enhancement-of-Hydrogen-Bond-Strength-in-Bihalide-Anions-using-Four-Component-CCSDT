def generate_xyz_files():
   
    base_z_1 = 1.56945000
    base_z_2 = -1.56945000
    base_z_H = 0.000000  

    for i in range(20):
        z_shift = i * 0.01 
        new_z_H = base_z_H + z_shift  
        new_z_1 = base_z_1 - 1.0078*(new_z_H / 70.9000)
        new_z_2 = base_z_2 - 1.0078*(new_z_H / 70.9000)

        
        content = f"""3
ClHCl
Cl 0.00000 0.00000 {new_z_1}
Cl 0.00000 0.00000 {new_z_2}
H  0.00000 0.00000 {new_z_H}
"""
       
        with open(f"ClHCl_Q3_{i+1}.xyz", "w") as f:
            f.write(content)

generate_xyz_files()
print("Four-component coffe time!")
