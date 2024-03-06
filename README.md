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
