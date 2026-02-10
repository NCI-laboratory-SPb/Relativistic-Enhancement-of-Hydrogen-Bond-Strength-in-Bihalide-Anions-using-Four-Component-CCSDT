def generate_xyz_files():
    
    base_y_1 = 0.000000
    base_y_2 = 0.000000
    base_y_H = 0.000000  

    for i in range(20):
        y_shift = i * 0.1  
        new_y_H = base_y_H + y_shift  
        new_y_1 = base_y_1 - 1.0078*(new_y_H / 70.9000)
        new_y_2 = base_y_2 - 1.0078*(new_y_H / 70.9000)

        
        content = f"""3
ClHCl
Cl  0.00000 {new_y_1} 1.56945000
Cl  0.00000 {new_y_2} -1.56945000
H   0.00000 {new_y_H} 0.00000
"""
        
        with open(f"ClHCl_Q2_{i+1}.xyz", "w") as f:
            f.write(content)

generate_xyz_files()
print("Four-component coffe time!")
