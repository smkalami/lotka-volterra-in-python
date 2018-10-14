import numpy as np
import matplotlib.pyplot as plt

# Lotka-Volterra Model Parameters
alpha = 1.1
beta = 0.4
gamma = 0.4
delta = 0.1

# Sample Time
dt = 0.01

# Simulation Time
N = 5000
t = np.arange(N)*dt

# Initial Values
x0 = 20
y0 = 5

# Dynamics of The Model
def f(x, y):
    xdot = alpha*x - beta*x*y
    ydot = delta*x*y - gamma*y
    return xdot, ydot

# State Transition using Runge-Kutta Method
def next(x, y):
    xdot1, ydot1 = f(x, y)
    xdot2, ydot2 = f(x + xdot1*dt/2, y + ydot1*dt/2)
    xdot3, ydot3 = f(x + xdot2*dt/2, y + ydot2*dt/2)
    xdot4, ydot4 = f(x + xdot3*dt, y + ydot3*dt)
    xnew = x + (xdot1 + 2*xdot2 + 2*xdot3 + xdot4)*dt/6
    ynew = y + (ydot1 + 2*ydot2 + 2*ydot3 + ydot4)*dt/6
    return xnew, ynew

# Simulation Loop
x = np.zeros(N)
y = np.zeros(N)
x[0] = x0
y[0] = y0
for k in range(N-1):
    x[k+1], y[k+1] = next(x[k], y[k])

# Plot Results
plt.subplot(1,2,1)
plt.plot(t,x, label='Prey', linewidth=1)
plt.plot(t,y, label='Predator', linewidth=1)
plt.grid()
plt.legend(loc = 'upper right')
plt.xlabel('Time')

plt.subplot(1,2,2)
plt.plot(x,y, linewidth=1)
plt.grid()
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Phase Portrait')

plt.show()
