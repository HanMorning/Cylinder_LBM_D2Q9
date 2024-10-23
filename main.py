import matplotlib.pyplot as plt
import numpy as np

"""
Python Program
Lattice Boltzmann Method - D2Q9
Fluid Flow Past a Fixed Cylinder
Han
"""

if __name__ == "__main__":
    # essential parameters: Fluid domain
    MAX_X = 400
    MAX_Y = 100
    # D2Q9 model
    LATTICE_NUM = 9
    CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    WEIGHTS = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
    OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    # cylinder
    POSITION_OX = 70
    POSITION_OY = 50
    RADIUS = 20
    x, y = np.meshgrid(range(MAX_X), range(MAX_Y))
    cylinder = (x - POSITION_OX) ** 2 + (y - POSITION_OY) ** 2 <= RADIUS ** 2
    # fluid parameters
    REYNOLDS = 200
    U_MAX = 0.1
    kinematic_viscosity = U_MAX * 2 * RADIUS / REYNOLDS
    relaxation_time = 3.0 * kinematic_viscosity + 0.5
    print(f"Reynolds number = {REYNOLDS}\nrelaxation time = {relaxation_time}")
    # main loop and plot-save
    MAX_STEP = 40001
    OUTPUT_STEP = 5000
    PICTURE_NUM = 1

    # initial Conditions
    rho = np.ones([MAX_Y, MAX_X])
    ux, uy = np.zeros([MAX_Y, MAX_X]), np.zeros([MAX_Y, MAX_X])
    ux[:, 0], ux[:, -1] = U_MAX, U_MAX
    F = np.zeros([MAX_Y, MAX_X, LATTICE_NUM])
    for i, cx, cy, w in zip(range(LATTICE_NUM), CX, CY, WEIGHTS):
        F[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy)
                                + 9 * (cx * ux + cy * uy) ** 2 / 2
                                - 3 * (ux ** 2 + uy ** 2) / 2)

    for step in range(MAX_STEP):
        print(step)

        # periodic boundary condition
        F[:, 0, [1, 5, 8]] = F[:, -1, [1, 5, 8]]
        F[:, -1, [3, 6, 7]] = F[:, 0, [3, 6, 7]]
        F[0, :, [2, 5, 6]] = F[-1, :, [2, 5, 6]]
        F[-1, :, [4, 7, 8]] = F[0, :, [4, 7, 8]]

        # stream
        for i, cx, cy in zip(range(LATTICE_NUM), CX, CY):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # cylinder boundary
        cylinderF = F
        for i in range(1, LATTICE_NUM):
            incoming_particles = cylinder & (np.roll(cylinder, -CX[i], axis=1) &
                                             np.roll(cylinder, -CY[i], axis=0) == False)
            cylinderF[incoming_particles, i] = F[incoming_particles, OPP[i]]

        # Fluid parameters
        rho = np.sum(F, 2)
        ux = np.sum(F * CX, 2) / rho
        uy = np.sum(F * CY, 2) / rho
        F = cylinderF
        ux[cylinder] = 0
        uy[cylinder] = 0

        # collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(LATTICE_NUM), CX, CY, WEIGHTS):
            Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy)
                                      + 9 * (cx * ux + cy * uy) ** 2 / 2
                                      - 3 * (ux ** 2 + uy ** 2) / 2)
        F += - (1 / relaxation_time) * (F - Feq)

        # inflow and outflow boundary condition
        ux[:, 0], ux[:, -1] = U_MAX, U_MAX
        for i, cx, w in zip(range(LATTICE_NUM), CX, WEIGHTS):
            F[:, 0, i] = rho[:, 0] * w * (1 + 3 * (cx * ux[:, 0])
                                          + 9 * (cx * ux[:, 0]) ** 2 / 2
                                          - 3 * (ux[:, 0] ** 2) / 2)
            F[:, -1, i] = rho[:, -1] * w * (1 + 3 * (cx * ux[:, -1])
                                            + 9 * (cx * ux[:, -1]) ** 2 / 2
                                            - 3 * (ux[:, -1] ** 2) / 2)

        # export vorticity and streamlines pictures
        if step % OUTPUT_STEP == 0:
            vorticity = (
                    (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
                    - (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0))
            )
            vorticity[cylinder] = np.nan
            plt.imshow(vorticity, cmap="bwr", origin="lower", vmin=-0.02, vmax=0.02)
            plt.gca().add_patch(plt.Circle((POSITION_OX, POSITION_OY), RADIUS, color="black"))

            Y, X = np.mgrid[0:MAX_Y, 0:MAX_X]
            speed = np.sqrt(ux ** 2 + uy ** 2)
            plt.streamplot(X, Y, ux, uy, color=speed, linewidth=1, cmap='cool')

            plt.title(f'Vorticity and Streamlines at Step {step}')
            plt.savefig(f"Lattice-Boltzmann-{PICTURE_NUM}")
            plt.pause(0.01)
            plt.cla()
            PICTURE_NUM = PICTURE_NUM + 1
