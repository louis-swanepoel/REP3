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

sol = integrate.solve_ivp(ode, [0, 1814400], [1.515*10**11, 5, 0.6, 4, 0, 2], t_eval = np.linspace(0, 1814400, 10000))

x_array = []
y_array = []
z_array = []

for i in range(0, len(sol.y[0])):
    x = sol.y[0][i] * math.cos(sol.y[4][i]) * math.cos(sol.y[2][i])
    x_array.append(x)
    
    y = sol.y[0][i] * math.cos(sol.y[4][i]) * math.sin(sol.y[2][i])
    y_array.append(y)
    
    z = sol.y[0][i] * math.sin(sol.y[4][i])
    z_array.append(z)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot(x_array, y_array, z_array)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot(x_values_list,
        y_values_list,
        z_values_list)
