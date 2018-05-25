"""
Author: Rohan Ramasamy
Date: 22/05/18

This file contains a 1D spherically symmetric solver to obtain the motion of ions in a negative potential well
"""


import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import *
from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import *
from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import *
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import PICParticle
from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.abe_collison_model import AbeCoulombCollisionModel

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle, CoulombCollision
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess

from plasma_physics.pysrc.utils.physical_constants  import PhysicalConstants


def run_1d_electrostatic_well(radius, num_particles=int(1e3)):
    """
    Function to run simulation - for this simulation, in a spherically symmetric potential well

    radius: size of potential well
    """
    # Define particles
    number_density = 1e8
    charge = 1.602e-19
    weight = int(number_density / num_particles)
    pic_particle = PICParticle(3.344496935079999e-27 * weight, charge * weight, np.asarray([radius, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.0]))
    collision_particle = ChargedParticle(3.344496935079999e-27 * weight, charge * weight)

    # Define fields
    electron_charge_density = 1e20 * PhysicalConstants.electron_charge
    def e_field(x):
        E_field = np.zeros(x.shape)
        for i in range(x.shape[0]):
            r = magnitude(x[i])
            if r <= radius:
                e_field = electron_charge_density * r / (3.0 * PhysicalConstants.epsilon_0) 
            else:
                e_field = electron_charge_density * radius ** 3 / (3.0 * PhysicalConstants.epsilon_0 * r ** 2)

            e_field *= -normalise(x[i])
            E_field[i, :] = e_field
        
        return E_field

    # Define time step and final time
    total_V = radius ** 3 * electron_charge_density
    total_V /= 2.0 * PhysicalConstants.epsilon_0 * radius
    energy = total_V * pic_particle.charge
    velocity = np.sqrt(2 * energy / pic_particle.mass)
    dt = 0.01 * radius / velocity
    final_time = 50.0 * radius / velocity
    num_steps = int(final_time / dt)

    # Define collision model
    temperature = 298.3
    thermal_velocity = np.sqrt(PhysicalConstants.boltzmann_constant * temperature / pic_particle.mass)
    c = CoulombCollision(collision_particle, collision_particle, 1.0, velocity)
    debye_length = PhysicalConstants.epsilon_0 * temperature
    debye_length /= number_density * PhysicalConstants.electron_charge ** 2
    debye_length = np.sqrt(debye_length)
    coulomb_logarithm = np.log(debye_length / c.b_90)
    r = RelaxationProcess(c)
    v_K = r.kinetic_loss_stationary_frequency(number_density, temperature, velocity)
    print("Thermal velocity: {}".format(thermal_velocity / 3e8))
    print("Peak velocity: {}".format(velocity / 3e8))
    print("Debye Length: {}".format(debye_length))
    print("Coulomb Logarithm: {}".format(coulomb_logarithm))
    print("Kinetic Loss Time: {}".format(1.0 / v_K))
    print("Simulation Time Step: {}".format(dt))
    assert dt < 0.01 * 1.0 / v_K
    collision_model = AbeCoulombCollisionModel(num_particles, collision_particle, weight, coulomb_logarithm=coulomb_logarithm)

    # Set up initial conditions
    np.random.seed(1)
    X = np.zeros((num_particles, 3))
    for i in range(num_particles):
        x_theta = np.random.uniform(0.0, 2 * np.pi)
        y_theta = np.random.uniform(0.0, 2 * np.pi)
        z_theta = np.random.uniform(0.0, 2 * np.pi)

        X[i, :] = rotate_3d(np.asarray([1.0, 0.0, 0.0]), np.asarray([x_theta, y_theta, z_theta]))
    V = np.random.normal(0.0, thermal_velocity, size=X.shape)
    V = np.zeros(X.shape)
    
    # Run simulation
    times = np.linspace(0.0, dt * num_steps, num_steps)
    positions = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    velocities = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    for i, t in enumerate(times):
        # Set first position and velocity
        if i == 0:
            positions[i, :, :] = X
            velocities[i, :, :] = V
            continue

        # Update position and velocity due to field
        dt = times[i] - times[i - 1]
        x = X + V + 0.5 * dt if i == 1 else X + V * dt
        E = e_field(x)
        v = V + E * pic_particle.charge / pic_particle.mass * dt

        # Update velocity due to collisions
        new_v = collision_model.single_time_step(v, dt)

        positions[i, :, :] = x
        velocities[i, :, :] = new_v
        X = x
        V = new_v

        print("Timestep: {}".format(i))

    return times, positions, velocities

if __name__ == '__main__':
    radius = 1.0
    times, positions, velocities = run_1d_electrostatic_well(radius)

    absolute_velocities = np.sqrt(velocities[:, :, 0] ** 2 + velocities[:, :, 1] ** 2 + velocities[:, :, 2] ** 2)

    plt.figure()

    plt.plot(times, absolute_velocities[:, 0])

    plt.show()

    # Plot 3D motion
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot('111', projection='3d')
    for i in range(10):
        x = positions[:, i, 0].flatten()
        y = positions[:, i, 1].flatten()
        z = positions[:, i, 2].flatten()
            
        ax.plot(x, y, z, label='particle {}'.format(1))
    
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Analytic and Numerical Particle Motion")
    plt.show()

