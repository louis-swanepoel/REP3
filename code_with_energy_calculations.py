
import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy import integrate

mass_of_jwt = 6000

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
time_start_orbit = 0
time_end_orbit = 1814400

r1i= 151500000000
r2i= 0
p1i= (0.00039708/180)*math.pi
p2i= 0.000000000009
q1i=(0.00039708/180)*math.pi
q2i= 0.00000009
#sol = integrate.solve_ivp(ode,[0,1814400], [151101100,185],[(3.9708)*10^-3,185.32],[(3.9708*10^-6), 185], t_eval = np.linspace(0, 1814400, 10000))
  
def integration(time_end_orbit,time_start_orbit,ode,r1i,r2i,p1i,p2i,q1i, q2i):
    
    number_of_integrations = 10
    
    sol = integrate.solve_ivp(ode,[time_start_orbit,time_end_orbit], [r1i,r2i,p1i,p2i,q1i, q2i], t_eval = np.linspace(time_start_orbit, time_end_orbit, number_of_integrations))    

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

time_in_a_day = 86400

for time_end_orbit in range(0,time_end_orbit,time_in_a_day):
    
    integration(time_end_orbit,time_start_orbit,ode,r1i,r2i,p1i,p2i,q1i, q2i)
    print(sol.y[0])

     
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









