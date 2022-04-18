# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib 
from scipy import integrate
from matplotlib.figure import Figure 


G = 6.674*10-11
m1 = 1.989*10**30
m2 = 5.972*10**24
mass_of_jwt = 6000
time_start_orbit = 0
time_end_orbit = 21
delta_vee = 150

# Functions to define the initial conditions for certain orbits outlined by NASA
def orbit1():
    
    # Initial conditions for orbit 1
    r1i = -4173087+151*10**9
    r2i = 5523
    p1i = math.tan(-6944925/r1i) 
    p2i = math.tan(-8120/r1i)
    q1i = math.tan(-538178/r1i) 
    q2i = math.tan(-972/r1i)
    return r1i, r2i, p1i, p2i, q1i, q2i

def orbit2():
    
    # Initial conditions for orbit 2
    r1i = 158222184+151*10**9
    r2i = -2048
    p1i = math.tan(35826901/r1i) 
    p2i = math.tan(-1/r1i)
    q1i = math.tan(-9272235/r1i)
    q2i = math.tan(-73/r1i)
    return r1i, r2i, p1i, p2i, q1i, q2i

def orbit3():
    
    # Initial conditions for orbit 2
    r1i = 1327098182+151*10**9
    r2i = 125
    p1i = math.tan(280369000/r1i) 
    p2i = math.tan( 144/r1i)
    q1i = math.tan(9696150/r1i)
    q2i = math.tan(46/r1i)
    return r1i, r2i, p1i, p2i, q1i, q2i


def orbit_trial():
    # Distance r
    r1i= 151*10**9
    
    # r dot 
    r2i= 200
    
    # Angle theta 
    p1i= -2.3725*10**-7
    
    # Theta dot
    p2i= 6.63*10**2
    
    # Angle phi
    q1i= -2.3765*10**-7
    
    # Phi dot 
    q2i= -4.83*10**-2
    return r1i, r2i, p1i, p2i, q1i, q2i



# Function using solve`_ivp in scipy to solve the set of differential equations found 
## using the Lagrange equations and returning the solutions as an array called variable sol
def integration(time_end_orbit, r1i, r2i, p1i, p2i, q1i, q2i):
    
    time_end_orbit = time_end_orbit
    
    def ode(t, r):
        
        r1, r2, p1, p2, q1, q2 = r
        
        r1_dot = r2
        r2_dot = r1 * ((p2**2) + ((math.sin(p1))**2 * (q2**2))) + ( G * (m1 - m2)/(r1**2) )  
        p1_dot = p2
        p2_dot = ( (math.sin(p1)) * (math.cos(p1)) * (q2**2) ) - ( (2 * r2 * p2)/r1 )
        q1_dot = q2
        q2_dot = - ( (( 2 * r2 * q2 )/r1) + ( 2 * ( (math.cos(p1))/(math.sin(p1)) ) * p2 * q2 ) )
        
        return r1_dot, r2_dot, p1_dot, p2_dot, q1_dot, q2_dot
    
    number_of_integrations = 400
    
    sol = integrate.solve_ivp(ode,[0, time_end_orbit], [r1i,r2i,p1i,p2i,q1i,q2i], t_eval = np.linspace(0, 
        time_end_orbit, number_of_integrations))    

    return sol 

# Function to change the spherical coordinate system to a cartesian coordinate system    
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


def graph(orbit):
    
    r1i, r2i, p1i, p2i, q1i, q2i = orbit
        
    sol = (integration(time_end_orbit, r1i, r2i, p1i, p2i, q1i, q2i) ) 
    
    x_values_list , y_values_list , z_values_list = change_to_cartesian(sol)  
    
    # Graphing
    plt.figure(figsize=(6, 6), dpi=100)
    
    ax1 = plt.axes(projection = '3d')
    ax1.plot( y_values_list, x_values_list,z_values_list)
           
    figure2 = plt.figure(2,figsize=(6, 6), dpi=100)
    ax2 = plt.axes(projection = '3d')
    ax2.set_title('21 days in orbit 3')
    ax2.set_xlabel("Theta (Rad)")
    ax2.set_ylabel("r (metres)")
    ax2.set_zlabel("Phi (Rad)")
    ax2.plot(sol.y[2],
            sol.y[0],
            sol.y[4])
    
    plt.show()

def deltaV(orbit):
        
    r1i, r2i, p1i, p2i, q1i, q2i = orbit
    sol = (integration(time_end_orbit, r1i, r2i, p1i, p2i, q1i, q2i) )
    x_values_list , y_values_list , z_values_list = change_to_cartesian(sol)
    
    # Calculate the potential energy
    distance_from_origin = math.sqrt(((x_values_list[-1] - x_values_list[0]) ** 2) + ((y_values_list[-1] - y_values_list[0]) ** 2) + ((z_values_list[-1] - z_values_list[0]) ** 2))
    U = -G*((m1*mass_of_jwt/distance_from_origin)+(m2*mass_of_jwt/(15*10**5)))
    # Calculate the kinetic energy
    current_velocity= math.sqrt(sol.y[1, -1] ** 2) + (sol.y[3, -1] ** 2) + (sol.y[5, -1] ** 2)
    initial_velocit = math.sqrt(r2i**2 +p2i**2 + q2i**2)
    delta_vi = initial_velocit - current_velocity
    
    return delta_vi,U


delta_vi,U = deltaV(orbit3())
graph(orbit1())
print("delta v for single manouvre:" , delta_vi)
print('U:', U)

