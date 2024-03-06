# %%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# %%
class MassSpringDamper:

    def __init__(self, m, k, c):
        self.m = m
        self.c = c
        self.k = k
        self.f = lambda t: 0
        self.A = np.array([[0, 1],
                           [-self.k/self.m, -self.c/self.m]])
        self.B = np.array([[0],
                           [1/self.m]])
        
    
    def set_force(self, f) -> None:
        self.f = f
    
    def calc_force(self, t):
        return np.array([[self.f(t)]])

    def ode(self, t: float, x:np.ndarray):
        x = x.reshape(2,1)
        x_dot = (self.A @ x) + (self.B @ self.calc_force(t))
        return x_dot.ravel()
    

# %% [markdown]
"""
Now integrating this with a non-zero initial condition of $\mathbf{x} = [5, 0]^T$, 
with a system of $m = 1 kg, k = 1 N/m, c = 0.5 Ns/m $, with no external
force, $f = 0$, we get the following response
"""

# %%
# Time
dt = 1e-3
t_start = 0
t_end = 10
t = np.arange(t_start, t_end, dt)
msd = MassSpringDamper(1,0.8,0.5)
x0 = np.array([5,0])
sol = integrate.solve_ivp(
    msd.ode,
    (t_start,t_end),
    x0,
    args=(),
    t_eval=t,
    rtol = 1e-6,
    atol=1e-6,
    method='RK45')

sol_x = sol.y
pos = sol_x[0,:]
vel = sol_x[1,:]

# %%
# Plotting
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

fig, ax = plt.subplots(2,1)

ax[0].set_ylabel(r'Position, $x(t)$, [m]')
ax[0].set_xlabel(r'$t$ [s]')
ax[0].plot(t,pos)

ax[1].set_ylabel(r'Velocity, $\dot{x}(t)$, [m/s]')
ax[1].set_xlabel(r'$t$ [s]')
ax[1].plot(t,vel)

fig.tight_layout()
plt.show()

# %% [markdown]
"""
Now with an external force $f = A\sin(\omega t)$ with $A=1N$ and 
a frequency of 1 Hz or $\omega = 2\pi \text{rad}/s$
"""


# %%
# Forcing function f(t) = A sin(wt)
f = lambda t: 1 * np.sin(2*np.pi*t)
# Change function in mass spring damper
msd.set_force(f)
# Reintegrate
sol = integrate.solve_ivp(
    msd.ode,
    (t_start,t_end),
    x0,
    args=(),
    t_eval=t,
    rtol = 1e-6,
    atol=1e-6,
    method='RK45')

sol_x = sol.y
pos = sol_x[0,:]
vel = sol_x[1,:]
# Plotting
# Plotting parameters
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

fig, ax = plt.subplots(2,1)

ax[0].set_ylabel(r'Position, $x(t)$, [m]')
ax[0].set_xlabel(r'$t$ [s]')
ax[0].plot(t,pos)

ax[1].set_ylabel(r'Velocity, $\dot{x}(t)$, [m/s]')
ax[1].set_xlabel(r'$t$ [s]')
ax[1].plot(t,vel)

fig.tight_layout()
plt.show()

# %% [markdown]
"""
## Generating Sensor Measurements
I'll set the sampling rate of both the accelerometer and 
position sensor to 100Hz. Moreover, I set the standard deviation of 
the position measurement to 1cm or 0.01m, which seems reasonable 
for a position value. 

For the accelerometer, it is less obvious what this might be. Using 
a datasheet for an average accelerometer, with $ \pm 4 $ g, 10-bit resolution
we have the lowest significant bit LSB representing $0.077\frac{m}{s^2}$.
The datasheet lists an LSB RMS Noise of 0.75, assuming a mean of zero I
set this uncertainty to $0.06\frac{m}{s^2}$.  
"""

# %%
def sensor_measurement(t:float,x:np.ndarray,msd:MassSpringDamper):
    accelerometer = []
    Q = 0.06
    position = []
    R = 0.01
    sensor_time = []
    # Was integrated with dt = 1e-3
    # For 100 Hz then must take every 10th sample
    for i in range(0,len(t),10):
        ddot_r_meas = msd.ode(t,x[:,i]) + Q*np.random.randn()
        accelerometer.append(ddot_r_meas[0])
        pos_meas = x[0,i] + R*np.random.randn()
        position.append(pos_meas)
        sensor_time.append(t[i])
    return sensor_time, position, accelerometer

time_data, pos_data, acc_data = sensor_measurement(t,sol_x,msd)


fig, ax = plt.subplots(2,1)

ax[0].set_ylabel(r'Position, $x(t)$, [m]')
ax[0].set_xlabel(r'$t$ [s]')
ax[0].plot(t,pos)
ax[0].scatter(time_data,pos_data, label = "Measured", color='C2',s=0.2)

ax[1].set_ylabel(r'Acceleration, $\ddot{x}(t)$, [m/s^2]')
ax[1].set_xlabel(r'$t$ [s]')
ax[1].scatter(time_data,acc_data,s=0.5)

fig.tight_layout()
plt.show()

# %% [markdown]
"""
## Discretization
Using the fact that $\ddot{r}(t) = u^{\text{acc}}(t) = 
a(t) + w(t)$. We can then rewrite our initial state space equation
as
$$
\dot{\mathbf{x}}(t) = 
\begin{bmatrix}
0 & 1 \\
0 & 0 \\
\end{bmatrix}\mathbf{x}(t) + 
\begin{bmatrix}
0 \\
1 \\
\end{bmatrix}a(t)
+ 
\begin{bmatrix}
0 \\
1 \\
\end{bmatrix}w(t)
$$
$$
\dot{\mathbf{x}}(t) = \mathbf{Ax}(t) + \mathbf{B}a(t) + \mathbf{L}w(t) \tag{1}
$$
Similarly, our position measurement can be represented as
$$
y(t) = \begin{bmatrix}
1 & 0 \\
\end{bmatrix}\mathbf{x}(t) + v(t)
$$
$$
y(t) = \mathbf{Cx}(t) + v(t) \tag{2}
$$
Taking the Laplace Transform of Equ. 1, we have that 
$$
\mathbf{x}(s) = (s -\mathbf{1A})^{-1}\mathbf{x}_0 + (s -\mathbf{1A})^{-1}\mathbf{B}a(s) + (s -\mathbf{1A})^{-1}\mathbf{L}w(s) \tag{3}
$$
The solution to this is then 
$$
x(t) = e^{\mathbf{A}(t-t_0)}\mathbf{x}(t_0) + 
        \int_{t_0}^{t}e^{\mathbf{A}(t-\tau)}\mathbf{B}a(\tau)d\tau +
        \int_{t_0}^{t}e^{\mathbf{A}(t-\tau)}\mathbf{L}w(\tau)d\tau
$$
Then through a zero-order hold, assuming the input is constant over
a time interval, $ T = t_{k}- t_{k-1} $, we have that 
$$
\mathbf{x}_k = \underbrace{e^{\mathbf{A}T}}_{\mathbf{A_{k-1}}}\mathbf{x}_{k-1} +
    \underbrace{\int_{0}^{T}e^{\mathbf{A}\tau}d\tau \mathbf{B}}_{\mathbf{B}_{k-1}}a_{k-1}
    + \underbrace{\int_{0}^{T}e^{\mathbf{A}\tau}d\tau \mathbf{L}w_{k-1}}_{\mathbf{w}_{k-1}} \tag{4}
$$
where $\mathbf{w}_{k-1}$ is defined as
$$
\mathbf{w}_{k-1} \sim \mathcal{N}(\mathbf{0},\mathbf{Q}_{k-1}) \tag{5}
$$
Using Van Loan's Method as detailed in [2,3], we can write
the process noise $\mathbf{Q}_{k-1}$ as
$$
\mathbf{Q}_{k-1} = \int_{0}^{T}e^{\mathbf{A}\tau}\mathbf{Q}e^{\mathbf{A^{\top}}\tau}d\tau
    = \mathbf{A}_{k-1}(\mathbf{A}_{k-1}^{-1}\mathbf{Q}_{k-1})
$$
Explicitly calculating $\mathbf{A}_{k-1}$ 
$$
\mathbf{A}_{k-1} = e^{\mathbf{A}\tau} = \mathbf{1} + \mathbf{A}T + \frac{(\mathbf{A}T)^2}{2!} + \ldots
$$
For the $\mathbf{A}$ defined in Equ. 1, we have that $\mathbf{A}^2 = 0$ thus
$$
\mathbf{A}_{k-1} = 
\begin{bmatrix}
1 & T \\
0 & 1\\
\end{bmatrix}
$$
Similarly, for $\mathbf{B}_{k-1}$ 
$$
\begin{align*}
\mathbf{B}_{k-1} &= \int_{0}^{T}e^{\mathbf{A}\tau}d\tau \mathbf{B} \\
&=  (\int_{0}^{T}\begin{bmatrix}
1 & \tau \\
0 & 1\\
\end{bmatrix})\mathbf{B} \\
&= \begin{bmatrix}
T & \frac{1}{2}T^2 \\
0 & T\\
\end{bmatrix}
\begin{bmatrix}
0 \\
1 \\
\end{bmatrix} \\
&= 
\begin{bmatrix}
\frac{1}{2}T^2 \\
T \\
\end{bmatrix}
\end{align*}
$$
Similarly, 
$$
\begin{align*}
\mathbf{w}_{k-1} &= \int_{0}^{T}e^{\mathbf{A}\tau}d\tau \mathbf{L}w_{k-1} \\
&= \begin{bmatrix}
\frac{1}{2}T^2 \\
T \\
\end{bmatrix}w_{k-1}
\end{align*}
$$
Finally, the discretized process model can be written as
$$
\mathbf{x}_k = 
\begin{bmatrix}
1 & T \\
0 & 1\\
\end{bmatrix}\mathbf{x}_{k-1} +
\begin{bmatrix}
\frac{1}{2}T^2 \\
T \\
\end{bmatrix}a_{k-1} + 
\begin{bmatrix}
\frac{1}{2}T^2 \\
T \\
\end{bmatrix}w_{k-1}
$$
For measurement model, the discretized form is much easier to write with 
$$
y_k = \begin{bmatrix} 1 & 0 \end{bmatrix}\mathbf{x}_k + v_k
$$
"""
# %%
class KalmanFilter():
    def __init__(self,A,B,C,Q,R):
        self.A = A
# %% [markdown]
"""
## Bibliography
[1] https://www.analog.com/media/en/technical-documentation/data-sheets/adxl345.pdf \
[2] https://en.wikipedia.org/wiki/Discretization \
[3]  Charles Van Loan: Computing integrals involving the matrix exponential, IEEE Transactions on Automatic Control. 23 (3): 395–404, 1978
"""

# %%
