import numpy as np
from matplotlib import (pyplot as plt)

def Lorenz(xyz, *, s, r, b, dt, num_steps):

    """
    Simulate and plot the Lorenz attractor.

    Parameters
    ----------
    xyz: array-like, shape (3,)
        Initial condition for the Lorenz system [x0, y0, z0].
    s: float, optional
        Parameter of the Rossler system.
    r: float, optional
        Parameter of the Rossler system.
    b: float, optional
        Parameter of the Rossler system.
    dt: float, optional
        Time step for the simulation.
    num_steps: int, optional
        Number of steps for the simulation.
    """
    def Lorenz_derivatives(xyz, s, r, b):
        x, y, z = xyz
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return np.array([x_dot, y_dot, z_dot])

    # Initialize an array to store the results
    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = xyz  # Set initial values

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + Lorenz_derivatives(xyzs[i], s, r, b) * dt

    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(*xyzs.T, lw=0.6)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"Lorenz Attractor (s={s}, r={r}, b={b})")
    plt.show()


def Rossler(xyz, *, a, b, c, dt, num_steps):
    """
    Simulate and plot the Rossler attractor.

    Parameters
    ----------
    xyz: array-like, shape (3,)
        Initial condition for the Rossler system [x0, y0, z0].
    a: float, optional
        Parameter of the Rossler system.
    b: float, optional
        Parameter of the Rossler system.
    c: float, optional
        Parameter of the Rossler system.
    dt: float, optional
        Time step for the simulation.
    num_steps: int, optional
        Number of steps for the simulation.
    """

    def Rossler_derivatives(xyz, a, b, c):
        x, y, z = xyz
        x_dot = -y - z
        y_dot = x + a * y
        z_dot = b + z * (x - c)
        return np.array([x_dot, y_dot, z_dot])

    # Initialize an array to store the results
    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = xyz  # Set initial values

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + Rossler_derivatives(xyzs[i], a, b, c) * dt

    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(*xyzs.T, lw=0.6)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"Rössler Attractor (a={a}, b={b}, c={c})")
    plt.show()


#Lorenz example from wiki - parameters s=10, r=28, b=2.667
#Lorenz([0., 1., 1.05], s=10, r=28, b=2.667, dt=0.01, num_steps=10000)

#Rossler example from wiki - paparameters a=0.2, b=0.2, c=5.7
#Rossler([0., 1., 1.05], a=0.2, b=0.2, c=5.7, dt=0.01, num_steps=10000)
Rossler([0., 1., 1.05], a=0.1, b=0.1, c=14, dt=0.01, num_steps=10000)

