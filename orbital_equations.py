# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math as m
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import integrate, linalg, optimize #integration and odes, linear␣ 􏰀→algebra, optimization/root-finding
from scipy.integrate import solve_ivp 
from scipy.integrate import odeint 
from mpl_toolkits.mplot3d import Axes3D
import scipy


def ode(t, position ):
    
   
    [[r] , [p] , [q]] = position 
   
    
    
    
    r1_dot = r[1]
    r2_dot = r[0]*(((p[1])^2)+(m.sin(p[0]))^2)*((q[1])^2)+G*((m_1/(r[0])^2)-(m_2/(r[0])^2))
    p1_dot = p[1]
    p2_dot = m.sin(p[0])*m.cos(p[0])*((q[1])^2)-(2/r[0])*r[1]*p[1] 
    q1_dot = q[1]
    q2_dot = -((2/r[0])*r[1]*q[0]+2*m.cos(p[0])*p[1]*q[1])
    
    return [r1_dot,r2_dot,p1_dot, p2_dot, q1_dot, q2_dot]
 

G = 1*10^-10    
m_1 = 1*10^10
m_2 = 2*10^5


r_init = 161*10^6
p_init = 1*10^-2
q_init = 1*10^-2

sol = solve_ivp( ode , [0, 21], [r_init, p_init, q_init])

fig = plt.figure()
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot(sol.position[0, :],
        sol.position[1, :],
        sol.position[2, :])
ax.set_title("solve_ivp")


