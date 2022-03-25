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
    
    r_position = r2
    r_velocity = r1 * ((p2**2) + ((math.sin(p1))**2 * (q2**2))) + ( G * (m1 - m2)/(r1**2) )  
    p_position = p2
    p_velocity = ( (math.sin(p1)) * (math.cos(p1)) * (q2**2) ) - ( (2 * r2 * p2)/r1 )
    q_position = q2
    q_velocity = - ( (( 2 * r2 * q2 )/r1) + ( 2 * ( (math.cos(p1))/(math.sin(p1)) ) * p2 * q2 ) )
    
    return r_position, r_velocity, p_position, p_velocity, q_position, q_velocity

sol = integrate.solve_ivp(ode, [0, 1814400], [1500000, 1000, 0.9, 1000, 0.7, 1000], t_eval = np.linspace(0, 1814400, 10000))

print(sol.y)


fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot(sol.y[0,:],
        sol.y[2,:],
        sol.y[4,:])
