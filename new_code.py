# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:50:59 2022

@author: arthu
"""

import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy import integrate

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

sol = integrate.solve_ivp(ode, [0, 1814400], [np.random.randint(0,1000), np.random.randint(0,1000), np.random.randint(0,1000), np.random.randint(0,1000), np.random.randint(0,1000), np.random.randint(0,1000)], t_eval = np.linspace(0, 1814400, 10000))

print(sol)

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
    


fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot(x_values_list,
        y_values_list,
        z_values_list)
