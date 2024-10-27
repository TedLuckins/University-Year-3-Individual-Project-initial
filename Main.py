import numpy as np
from matplotlib import (pyplot as plt)


def Euler_Method(Deriv_Func, Initial_Conditions, num_steps, dt, *params):
    """
    :param Deriv_Func: function
            The function the computes the deriviatives
    :param Intital_Conditions: array-like, shape (3,)
            Initial values of the system (x, y, z)
    :param num_steps: int
            Number of time iterations
    :param dt: float
            Time step of each iteration
    :param params: tuple
            parameters of the system for derivation function
    :return: xyzs: ndarray, shape (num_steps + 1, 3)
            Array of (x, y, z) values at each time step
    """
    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = Initial_Conditions

    # Euler integration loop
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + Deriv_Func(xyzs[i], *params) * dt

    return xyzs


def Runge_Kutta_Method(Deriv_Func, Initial_Conditions, num_steps, dt, *params):
    """
    :param Deriv_Func: function
            The function the computes the deriviatives
    :param Intital_Conditions: array-like, shape (3,)
            Initial values of the system (x, y, z)
    :param num_steps: int
            Number of time iterations
    :param dt: float
            Time step of each iteration
    :param params: tuple
            parameters of the system for derivation function
    :return: xyzs: ndarray, shape (num_steps + 1, 3)
            Array of (x, y, z) values at each time step
    """
    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = Initial_Conditions

    #Runge-Kutta integration loop
    for i in range(num_steps):
        k1 = Deriv_Func(xyzs[i], *params) #Computes the derivative at current point (using Euler method)
        k2 = Deriv_Func(xyzs[i]+ 0.5 * k1 * dt, *params) #Computes the derivative at the midpoint, using k1
        k3 = Deriv_Func(xyzs[i]+ 0.5 * k2 * dt, *params) #COpmputes the derivative at another midpoint, using k2
        k4 = Deriv_Func(xyzs[i] + k3 * dt, *params) #Computes the derivative at the endpoint, using k3

        # Average the k increments with weight (1/6, 1/3, 1/3, 1/6) to update the solution
        xyzs[i + 1] = xyzs[i] + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0)

    return xyzs


def Lorenz(Integration_Func, xyz, *, s, r, b, dt, num_steps):
    """
    Simulate and plot the Lorenz attractor.

    param xyz: array-like, shape (3,)
            Initial condition for the Lorenz system [x0, y0, z0].
    param s: float, optional
            Parameter of the Lorenz system.
    param r: float, optional
            Parameter of the Lorenz system.
    param b: float, optional
            Parameter of the Lorenz system.
    param dt: float, optional
            Time step for the simulation.
    param num_steps: int, optional
            Number of steps for the simulation.
    """

    def Lorenz_derivatives(xyz, s, r, b):
        x, y, z = xyz
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return np.array([x_dot, y_dot, z_dot])

    if Integration_Func == "Euler":
        xyzs = Euler_Method(Lorenz_derivatives, xyz, num_steps, dt, s, r, b)

    if Integration_Func == "Runge-Kutta":
        xyzs = Runge_Kutta_Method(Lorenz_derivatives, xyz, num_steps, dt, s, r, b)


    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(*xyzs.T, lw=0.6)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"Lorenz Attractor (s={s}, r={r}, b={b})")
    plt.show()


def Rossler(Integration_Func, xyz, *, a, b, c, dt, num_steps):
    """
    Simulate and plot the Rossler attractor

    :param Integration_Func: String
            Integration method
    :param xyz: array-like, shape (3,)
            Initial condition for the Rossler system [x0, y0, z0].
    :param a: float, optional
            Parameter of the Rossler system.
    :param b: float, optional
            Parameter of the Rossler system.
    :param c: float, optional
            Parameter of the Rossler system.
    :param dt: float, optional
            Time step for the simulation.
    :param num_steps: int, optional
        Number of steps for the simulation.
    """

    def Rossler_derivatives(xyz, a, b, c):
        x, y, z = xyz
        x_dot = -y - z
        y_dot = x + a * y
        z_dot = b + z * (x - c)
        return np.array([x_dot, y_dot, z_dot])

    if Integration_Func == "Euler":
        xyzs = Euler_Method(Rossler_derivatives, xyz, num_steps, dt, a, b, c)

    if Integration_Func == "Runge-Kutta":
        xyzs = Runge_Kutta_Method(Rossler_derivatives, xyz, num_steps, dt, a, b, c)

    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(*xyzs.T, lw=0.6)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"Rössler Attractor (a={a}, b={b}, c={c})")
    plt.show()


# Lorenz example from wiki - parameters s=10, r=28, b=2.667
# Lorenz("Euler", [0., 1., 1.05], s=10, r=28, b=2.667, dt=0.01, num_steps=10000)
# Lorenz("Runge-Kutta", [0., 1., 1.05], s=10, r=28, b=2.667, dt=0.01, num_steps=10000)

# Rossler example from wiki - paparameters a=0.2, b=0.2, c=5.7
# Rossler("Euler", [0., 1., 1.05], a=0.2, b=0.2, c=5.7, dt=0.01, num_steps=10000)
Rossler("Euler", [0., 1., 1.05], a=0.1, b=0.1, c=14, dt=0.01, num_steps=10000)
Rossler("Runge-Kutta", [0., 1., 1.05], a=0.1, b=0.1, c=14, dt=0.01, num_steps=10000)

