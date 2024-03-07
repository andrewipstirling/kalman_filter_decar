# kalman_filter_decar 

## Mass Spring Damper System
### 1.
The equation of motion for the mass spring damper system is

$$
m \ddot{r}(t) + c\dot{r}(t) + kr(t) = f(t)
$$

We can write this in state space form, dropping the (t) notation for
brevity, as 

$$
\dot{\mathbf{x}} = 
\begin{bmatrix}
\dot{x} \\
\ddot{x}\\
\end{bmatrix} =
\begin{bmatrix}
0 & 1 \\
\frac{-k}{m} & \frac{-c}{m} \\
\end{bmatrix}
\begin{bmatrix}
x \\
\dot{x}\\
\end{bmatrix} + 
\begin{bmatrix}
0 \\
\frac{1}{m}\\
\end{bmatrix}
u
$$

Where $u = f(t)$ 

## Discretization
### 1.

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

### 2.
Taking the Laplace Transform of Equ. 1, we have that

$$
\mathbf{x}(s) = (s -\mathbf{1A})^{-1}\mathbf{x}_0 + (s -\mathbf{1A})^{-1}\mathbf{B}a(s) + (s -\mathbf{1A})^{-1}\mathbf{L}w(s) \tag{3}
$$

The solution to this is then

$$
x(t) = e^{\mathbf{A}(t-t_0)}\mathbf{x}(t_0) + \int_{t_0}^{t}e^{\mathbf{A}(t-\tau)}\mathbf{B}a(\tau)d\tau +\int_{t_0}^{t}e^{\mathbf{A}(t-\tau)}\mathbf{L}w(\tau)d\tau
$$

Then through a zero-order hold, assuming the input is constant over
a time interval, $T = t_{k}- t_{k-1}$, we have that

$$
\mathbf{x}_k = \underbrace{e^{\mathbf{A}T}}_{\mathbf{A_{k-1}}}\mathbf{x}_{k-1}+\underbrace{\int_{0}^{T}e^{\mathbf{A}\tau}d\tau \mathbf{B}}_{\mathbf{B}_{k-1}}a_{k-1}+\underbrace{\int_{0}^{T}e^{\mathbf{A}\tau}d\tau \mathbf{L}w_{k-1}}_{\mathbf{w}_{k-1}} \tag{4}
$$

where $\mathbf{w}_{k-1}$ is defined as

$$
\mathbf{w}_{k-1} \sim \mathcal{N}(\mathbf{0},\mathbf{Q}_{k-1}) \tag{5}
$$

Using Van Loan's Method as detailed in [2,3], we can write
the process noise $\mathbf{Q}_{k-1}$ as

$$
\mathbf{Q}_{k-1} = \int_{0}^{T}e^{\mathbf{A}\tau}\mathbf{Q}e^{\mathbf{A^{\top}}\tau}d\tau = \mathbf{A}_{k-1}(\mathbf{A}_{k-1}^{-1}\mathbf{Q}_{k-1})
$$

Explicitly calculating $\mathbf{A}_{k-1}$

$$\mathbf{A}_{k-1} = e^{\mathbf{A}\tau} = \mathbf{1} + \mathbf{A}T + \frac{(\mathbf{A}T)^2}{2!} + \ldots$$

For the $\mathbf{A}$ defined in Equ. 1, we have that $\mathbf{A}^2 = 0$ thus

$$\mathbf{A}_{k-1} =
\begin{bmatrix}1 & T \\
0 & 1\\
\end{bmatrix}$$

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

```math
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
\end{bmatrix} w_{k-1}
```

For the measurement model, the discretized form is equivalent to the continuous one with

$$
y_k = \begin{bmatrix}
1 & 0
\end{bmatrix} \mathbf{x}_k + v_k
$$
