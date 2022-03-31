
import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy import integrate

mass_of_jwt = 6000
time_start_orbit = 0
time_end_orbit = 10000

r1i= 151500000000
r2i= 0.03
p1i= 1*10^-10
p2i= 0.00000000000001
q1i= 1*10^-10
q2i= 0.00000000000001




#sol = integrate.solve_ivp(ode,[0,1814400], [151101100,185],[(3.9708)*10^-3,185.32],[(3.9708*10^-6), 185], t_eval = np.linspace(0, 1814400, 10000))
  
def integration(time_end_orbit, r1i, r2i, p1i, p2i, q1i, q2i):
    
    time_end_orbit = time_end_orbit
    
    def ode(t, r):
        
        
        
        G = 6.674*10-11
        m1 = 1.989*10**30
        m2 = 5.972*10**24
        
        r1, r2, p1, p2, q1, q2 = r
        
        r1_dot = r2
        r2_dot = r1 * ((p2**2) + ((math.sin(p1))**2 * (q2**2))) + ( G * (m1 - m2)/(r1**2) )  
        p1_dot = p2
        p2_dot = ( (math.sin(p1)) * (math.cos(p1)) * (q2**2) ) - ( (2 * r2 * p2)/r1 )
        q1_dot = q2
        q2_dot = - ( (( 2 * r2 * q2 )/r1) + ( 2 * ( (math.cos(p1))/(math.sin(p1)) ) * p2 * q2 ) )
        
        return r1_dot, r2_dot, p1_dot, p2_dot, q1_dot, q2_dot
    
    number_of_integrations = 320
    
    sol = integrate.solve_ivp(ode,[0, time_end_orbit], [r1i,r2i,p1i,p2i,q1i,q2i], t_eval = np.linspace(0, time_end_orbit, number_of_integrations))    

    return sol 
    
def change_to_cartesian(sol):
    
    x_values_list = []
    y_values_list = []
    z_values_list = []
    
    for i in range(0, len(sol.y[0])):
        x_value = sol.y[0, i] * math.cos(sol.y[2, i]) * math.cos(sol.y[4, i])
        x_values_list.append(x_value)
        
        y_value = sol.y[0, i] * math.cos(sol.y[2, i]) * math.sin(sol.y[4, i])
        y_values_list.append(y_value)
        
        z_value = sol.y[0, i] * math.sin(sol.y[2, i])
        z_values_list.append(z_value)
    return  x_values_list , y_values_list , z_values_list
def measurevariable(variable,final_value):
    
    
    time_span = range(0,final_value)
    
    r = []
    theta = []
    phi = []
    
    for variable in range(0,final_value):
        
        sol = integration(time_end_orbit,r1i,r2i,p1i,p2i,q1i, q2i)
        r_final = sol.y[0][9]
        theta_final = sol.y[2][9]
        phi_final = sol.y[4][9]
        
        r.append(r_final)
        theta.append(theta_final)
        phi.append(phi_final)
        
    
    
    
    fig , graph_variable_against_t = plt.subplots(3)
    
    graph_variable_against_t[0] = plt.axes()
    graph_variable_against_t[1] = plt.axes()
    graph_variable_against_t[2] = plt.axes()
    
    graph_variable_against_t[0].plot(r, time_span)    
    graph_variable_against_t[1].plot(theta, time_span)    
    graph_variable_against_t[2].plot(phi, time_span)
    
    
  
    ###### Optimal velocites for entry
    
# measurevariable(r2i, 320)
    
    ###### Optimal positions for entry
for p2i in []:
    
    sol = integration(time_end_orbit, r1i, r2i, p1i, p2i, q1i, q2i)
    
    graph_of_path_spherical = plt.figure('graph_of_path_spherical')
    graph_of_path_spherical = plt.axes(projection = '3d')
    graph_of_path_spherical.plot( sol.y[2], sol.y[0],sol.y[4])
    
    x_values_list , y_values_list , z_values_list = change_to_cartesian(sol)
    
    # graph_of_path_cartesian = plt.figure('graph_of_path_cartesian')
    # graph_of_path_cartesian = plt.axes(projection = '3d')
    # graph_of_path_cartesian.plot( y_values_list, x_values_list,z_values_list)
    plt.show()
# graph_r_against_t = plt.axes()
# graph_r_against_t.plot(r_final, np.linspace(0,320))





# print(y_values_list)
# print(z_values_list)


# method 2

# ax1 = plt.axes(projection = '3d')
# ax1.plot( y_values_list, x_values_list,z_values_list)
 
       

# method 2

# ax2 = plt.axes(projection = '3d')
# ax2.plot(sol.y[2],
#         sol.y[0],
#         sol.y[4])

# plt.show(ax2)

# method for individual plots

# ax3 = plt.axes()
# ax3.plot(np.linspace(time_start_orbit, time_end_orbit, 1000),sol.y[0])

# theta, phi = sol.y[2], sol.y[4]
# THETA, PHI = np.meshgrid(theta, phi)
# R = sol.y[0]
# X = R * np.sin(PHI) * np.cos(THETA)
# Y = R * np.sin(PHI) * np.sin(THETA)
# Z = R * np.cos(PHI)
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# plot = ax.plot()
# X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
# linewidth=0, antialiased=False, alpha=0.5)

# plt.show()
# ax.set_xlabel('')
# ax.set_ylable()
# ax.set_zlabel()


# Scatter Graph

# ax1.scatter3D(x_values_list,
#         y_values_list,
#         z_values_list)

plt.show()
# plt.savefig('plot.png')


#Calculate the kinetic energy
# velocity_squared = (sol.y[1, -1] ** 2) + (sol.y[3, -1] ** 2) + (sol.y[5, -1] ** 2)
# kinetic_evergy = 0.5 * mass_of_jwt * velocity_squared

# # print(kinetic_evergy)


# #Calculate the potential energy
# distance_from_origin = math.sqrt(((x_values_list[-1] - x_values_list[0]) ** 2) + ((y_values_list[-1] - y_values_list[0]) ** 2) + ((z_values_list[-1] - z_values_list[0]) ** 2))

# print(distance_from_origin)










