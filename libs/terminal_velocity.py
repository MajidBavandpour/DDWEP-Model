############################################################
# Developed by Majid Bavandpour(a) and Facundo Scordo(b)
# (a) Ph.D. Student @ UNR <majid.bavandpour@gmail.com>
# (b) Research Faculty @ UNR <fscordo@unr.edu>
# Summer 2025
#
############################################################
#
#
# The function here is used to calculate terminal velocity for each particle bin. 
#
#
############################################################


import numpy as np
import pandas as pd 
import sys, os



def calc(particle_diameter_csv_path, rho_air, rho_char, drag_coefficient_constant_method, constant_drag_coefficient, g, mu):


    d_names = ["Low_d_(mm)", "High_d_(mm)"]

    epsilon = 0.001

    df_particle_diameters = pd.read_csv(particle_diameter_csv_path)
    df_out = pd.DataFrame()

        
    for d_name in d_names:

        print(d_name)

        terminal_velocity = []
        drag_coefficient_list = []
        Re_list = []

        for row_index in range(len(df_particle_diameters)):

            print("\r", row_index, "/", len(df_particle_diameters), end='', flush=True)

            d = df_particle_diameters.iloc[row_index][d_name]/1000

            if not drag_coefficient_constant_method:

                vt_init = (g*d**2*(rho_char-rho_air))/(18*mu)

                diff = 100

                while diff>epsilon:

                    raynolds_number = (rho_air*vt_init*d)/mu

                    # print(raynolds_number)

                    if raynolds_number<=0.1:
                        # a = input("press any key 1 ....")
                        vt = vt_init
                        dr_coeff = 24/raynolds_number
                        break
                    
                    elif (raynolds_number>0.1 and raynolds_number<=1000):
                        # a = input("press any key 2 ....")

                        drag_coefficient = (24/raynolds_number)*(1+(0.15*np.power(raynolds_number, 0.687)))
                        vt = np.sqrt((4*g*d*(rho_char-rho_air))/(3*rho_air*drag_coefficient))
                        dr_coeff = drag_coefficient

                    elif raynolds_number>1000:
                        # a = input("press any key 3 ....")

                        vt = np.sqrt((4*g*d*(rho_char-rho_air))/(3*rho_air*constant_drag_coefficient))
                        dr_coeff = constant_drag_coefficient
                    
                    diff = np.abs(vt-vt_init)
                    vt_init = vt
            
            else:

                vt = np.sqrt((4*g*d*(rho_char-rho_air))/(3*rho_air*constant_drag_coefficient))
                dr_coeff = constant_drag_coefficient
                raynolds_number = 0



            terminal_velocity.append(vt)
            drag_coefficient_list.append(dr_coeff)
            Re_list.append(raynolds_number)


        
        df_out["vt_{}".format(d_name)] = terminal_velocity
        df_out["dr_{}".format(d_name)] = drag_coefficient_list
        df_out["re_{}".format(d_name)] = Re_list

        print("\n terminal velocity calculation :: Done!")
    

    # df_out.to_csv("data/terminal_velocity.csv")
    # sys.exit()


    return df_out

