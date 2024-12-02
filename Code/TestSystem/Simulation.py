import hoomd
import gsd.hoomd #hoomd
import mbuild as mb #pip
import numpy as np #hoomd
import matplotlib.pyplot as plt #pip
import glob #
import os #
import freud #pip
import time #hoomd
import scipy #hoomd
from input import * #pip
import warnings #
import random #

# Ignore DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

############### Functions #######################

# Create System from CSV data and Occupied Volume Fraction
def create_system(SpeciesList, box_length, seed, Time):
    print('-------Creating System-------')
    print("Seed", seed)
    # Sort the SpeciesList based on particle radius in descending order
    SpeciesList.sort(key=lambda x: x[4], reverse=True)
    # Debug: Print the contents of the SpeciesList DataFrame
    print(SpeciesList)

    # Define the system box
    box_length_mb = box_length / 10  # HOOMD uses angstrom, mbuild uses nm
    box = mb.Box(lengths=[box_length_mb, box_length_mb, box_length_mb])

    time=Time
    particles = []  # Init empty list to store particles
    n_compounds = []

    for species in SpeciesList:

        name, charge, mass, n, radius = species
        print('Adding Species:',species,time)
        # Calculate the maximum allowed position for the center of the particle
        max_position = box_length_mb - 2 * radius

        # Randomly generate positions within the box
        x = random.uniform(0, max_position)
        y = random.uniform(0, max_position)
        z = random.uniform(0, max_position)

        # Create the particle and add it to the list
        particle = mb.Particle(name=name, pos=[x, y, z], charge=charge, mass=mass, element=None, box=box)
        particles.append(particle)
        n_compounds.append(n)

    # Mbuild fill_box to create the system

    system = mb.fill_box(compound=particles, n_compounds=n_compounds, box=box, sidemax=box_length_mb+10, seed=seed)

    # Save the system as a .gsd file
    system.save("initial_PreRamp.gsd", overwrite=True)
    print('-------Done Creating System-------')
    return system

#Ramp the box from a large to small size and equilibrate
def Box_Ramp(snapshot_file, Box_Ramp_Factor, box_length,box_length_final, ramp_rate,ramp_dt, ramp_sim_steps, kT, Viscosity, EnergyScaling, dt, TotalTime, seed, time, LJCut, ramp_steps2fin, fast_trigger_rate, kappa):
    # Some print statements to verify that shit hasnt hit the fan
    print('-------Box Ramp-------')
    # Open the snapshot file
    gsd.hoomd.open(name=snapshot_file, mode='r')

    # Define Sim parameters
    gpu = hoomd.device.GPU()
    sim = hoomd.Simulation(device=gpu, seed=seed)
    sim.create_state_from_gsd(filename=snapshot_file)

    # Define LJ interactions between particles
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4), mode='xplor')

    # Define Yukawa interactions between particles
    yukawa = hoomd.md.pair.Yukawa(nlist=hoomd.md.nlist.Cell(buffer=0.4), mode='xplor')

    for i, species_vec_1 in enumerate(SpeciesList):
        for j, species_vec_2 in enumerate(SpeciesList):

            species1_name = species_vec_1[0]
            species2_name = species_vec_2[0]
            radius1 = species_vec_1[4]
            radius2 = species_vec_2[4]
            charge1 = species_vec_1[1]
            charge2 = species_vec_2[1]
            epsilon = InteractionList[i][j]
            sigma = (radius1 + radius2)

            lj.params[(species1_name, species2_name)] = dict(epsilon=epsilon*EnergyScaling, sigma=sigma)

            if  LJCut == True:
                lj.r_cut[(species1_name, species2_name)] = (2**(1/6))*sigma
            else:
                lj.r_cut[(species1_name, species2_name)] = (3*sigma)

            #print('LJCut for:',species1_name,species2_name)
            #print(lj.r_cut[(species1_name, species2_name)])

            # Set Yukawa parameters (You might need to adjust these values according to your needs)
            kappa = kappa # Calculated Kappa as inverse Debye Length
            epsilon_yukawa = charge1* charge2   # Scale Yukawa epsilon based on coulombs law and EnergyScaling

            yukawa.params[(species1_name, species2_name)] = dict(epsilon=epsilon_yukawa, kappa=kappa)

            # Setting cut-off radius for Yukawa (Adjust this value according to your needs)
            yukawa.r_cut[(species1_name, species2_name)] = lj.r_cut[(species1_name, species2_name)]


    # Brownian Dynamics
    brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=kT, default_gamma=1.0, default_gamma_r=(1.0, 1.0, 1.0))

    # Set custom gamma values for each species based on species_list
    for species_info in SpeciesList:
        species_name = species_info[0]
        species_radius = species_info[4]
        gamma = (6 * np.pi * Viscosity * species_radius)
        brownian.gamma[species_name] = gamma

    # Integrator
    integrator = hoomd.md.Integrator(dt=dt, methods=[brownian])

    # Add LJ potential to the integrator
    integrator.forces.append(lj)

    # Add LJ and Yukawa potentials to the integrator
    integrator.forces.extend([lj, yukawa])

    # Add integrator to the simulation
    sim.operations.integrator = integrator

    # Set random velocities and equilibrate
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

    # Monitoring Properties
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermodynamic_properties)

    # Define ramp to
    ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=ramp_steps2fin)

    # Define the start and stop boxes
    initial_box = sim.state.box
    print("Initial Box", initial_box)
    final_box = hoomd.Box(box_length_final,box_length_final,box_length_final,0,0,0)
    print("Final Box", final_box)
    box_resize_trigger = hoomd.trigger.Periodic(ramp_rate)

    box_resize = hoomd.update.BoxResize(box1=initial_box,
                                    box2=final_box,
                                    variant=ramp,
                                    trigger=box_resize_trigger)
    sim.operations.updaters.append(box_resize)

    sim.run(0)

    sim.run(1)
    print("--- Start Box Ramp ---")
    print("Start Time:",time.ctime())
    print("Seed:",seed)
    print("kT:",kT)
    print("EnergyScaling:",EnergyScaling)
    print("Eq_ScreeningLength:",kappa**-1)
    print("Eq_DOF:", thermodynamic_properties.degrees_of_freedom)
    print("Eq_KE:", thermodynamic_properties.kinetic_energy)
    print("Eq_PE:", thermodynamic_properties.potential_energy)
    print("Eq_Kin_Temp:", thermodynamic_properties.kinetic_temperature)
    print("Eq_Vol:", thermodynamic_properties.volume)
    print("Given BoxLength", box_length_final)
    print("Eq_Box_len:", np.power(thermodynamic_properties.volume, 1 / 3))

    logger = hoomd.logging.Logger()
    logger.add(thermodynamic_properties)
    logger.add(sim, quantities=['timestep', 'walltime'])

    # Running the Sim
    gsd_writer = hoomd.write.GSD(
        filename='initial_PostRamp.gsd',
        trigger=hoomd.trigger.Periodic(10000),
        mode='wb',
        dynamic=['property', 'momentum']
    )

    sim.operations.writers.append(gsd_writer)
    gsd_writer.logger = logger

    sim.run(ramp_sim_steps)
    print("--- Ramp Done ---")
    print("Post_DOF:", thermodynamic_properties.degrees_of_freedom)
    print("Post_ScreeningLength:",kappa**-1)
    print("Post_KE:", thermodynamic_properties.kinetic_energy)
    print("Post_PE:", thermodynamic_properties.potential_energy)
    print("Post_Kin_Temp:", thermodynamic_properties.kinetic_temperature)
    print("Post_Vol:", thermodynamic_properties.volume)
    print("Post_Box_len:", np.power(thermodynamic_properties.volume, 1 / 3))
    print("End Time:",time.ctime())
    print("--- End Of Min ---")
    print('------Done with Box Ramp--------')

# main MD simulation
def run_brownian_dynamics_simulation(snapshot_file, kT, Viscosity, EnergyScaling, dt, TotalTime, seed, time, LJCut, fast_trigger_rate,  slow_trigger_rate, switch_step, kappa):
    # Open the snapshot file
    gsd.hoomd.open(name=snapshot_file, mode='r')

    # Define Sim parametersslow_trigger_rate
    gpu = hoomd.device.GPU()
    sim = hoomd.Simulation(device=gpu, seed=seed)
    sim.create_state_from_gsd(filename=snapshot_file)

    # Define LJ interactions between particles
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4))

    # Define Yukawa interactions between particles
    yukawa = hoomd.md.pair.Yukawa(nlist=hoomd.md.nlist.Cell(buffer=0.4))

    for i, species_vec_1 in enumerate(SpeciesList):
        for j, species_vec_2 in enumerate(SpeciesList):


            species1_name = species_vec_1[0]
            species2_name = species_vec_2[0]
            radius1 = species_vec_1[4]
            radius2 = species_vec_2[4]
            charge1 = species_vec_1[1]
            charge2 = species_vec_2[1]
            epsilon = InteractionList[i][j]
            sigma = (radius1 + radius2) / 2
            lj.params[(species1_name, species2_name)] = dict(epsilon=epsilon*EnergyScaling, sigma=sigma)

            if  LJCut == True:
                lj.r_cut[(species1_name, species2_name)] = (2**(1/6))*sigma
            else:
                lj.r_cut[(species1_name, species2_name)] = (3*sigma)

            # Set Yukawa parameters (You might need to adjust these values according to your needs)
            kappa =  kappa # Calculated Kappa as inverse Debye Length
            epsilon_yukawa = charge1 * charge2   # Scale Yukawa epsilon based on coulombs law and EnergyScaling

            yukawa.params[(species1_name, species2_name)] = dict(epsilon=epsilon_yukawa, kappa=kappa)

            # Setting cut-off radius for Yukawa (Adjust this value according to your needs)
            yukawa.r_cut[(species1_name, species2_name)] = lj.r_cut[(species1_name, species2_name)]

    # Brownian Dynamics
    brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=kT, default_gamma=1.0, default_gamma_r=(1.0, 1.0, 1.0))

    # Set custom gamma values for each species based on species_list
    for species_info in SpeciesList:
        species_name = species_info[0]
        species_radius = species_info[4]
        gamma = (6 * np.pi * Viscosity * species_radius)
        brownian.gamma[species_name] = gamma

    # Integrator
    integrator = hoomd.md.Integrator(dt=dt, methods=[brownian])

    # Add LJ potential to the integrator
    integrator.forces.append(lj)

    # Add LJ and Yukawa potentials to the integrator
    integrator.forces.extend([lj, yukawa])

    # Add integrator to the simulation
    sim.operations.integrator = integrator

    # Set random velocities and equilibrate
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

    # Monitoring Properties
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermodynamic_properties)

    sim.run(0)

    sim.run(1)
    print("--- New Simulation ---")
    print("Start Time:",time.ctime())
    print("Seed:",seed)
    print("kT:",kT)
    print("EnergyScaling:",EnergyScaling)
    print("Eq_DOF:", thermodynamic_properties.degrees_of_freedom)
    print("Eq_KE:", thermodynamic_properties.kinetic_energy)
    print("Eq_PE:", thermodynamic_properties.potential_energy)
    print("Eq_Kin_Temp:", thermodynamic_properties.kinetic_temperature)
    print("Eq_Vol:", thermodynamic_properties.volume)
    print("Given BoxLength", box_length_final)
    print("Eq_Box_len:", np.power(thermodynamic_properties.volume, 1 / 3))

    logger = hoomd.logging.Logger()
    logger.add(thermodynamic_properties)
    logger.add(sim, quantities=['timestep', 'walltime'])

    # Create a slow writer with dynamic property and momentum
    gsd_writer = hoomd.write.GSD(
        filename='trajectory_slow.gsd',
        trigger=hoomd.trigger.Periodic(slow_trigger_rate),
        mode='wb',  # 'wb' to start a new file
        dynamic=['property', 'momentum']
    )
    gsd_writer.logger = logger  # Attach the logger to the writer

    # Add the writer to the simulation
    sim.operations.writers.append(gsd_writer)

    # Run the simulation with the slow writer for switch_step steps
    sim.run(switch_step)

    # Remove the slow writer and add the fast writer
    sim.operations.writers.remove(gsd_writer)
    gsd_writer = hoomd.write.GSD(
        filename='trajectory_fast.gsd',
        trigger=hoomd.trigger.Periodic(fast_trigger_rate),
        mode='ab',  # 'ab' to append to the existing file
        dynamic=['property', 'momentum']
    )
    gsd_writer.logger = logger  # Attach the logger to the writer
    sim.operations.writers.append(gsd_writer)

    # Continue running the simulation with the fast writer
    sim.run(TotalTime - switch_step)  # Run for the remaining steps

    print("--- EQ Done Running Simulation ---")
    print("Post_DOF:", thermodynamic_properties.degrees_of_freedom)
    print("Post_KE:", thermodynamic_properties.kinetic_energy)
    print("Post_PE:", thermodynamic_properties.potential_energy)
    print("Post_Kin_Temp:", thermodynamic_properties.kinetic_temperature)
    print("Post_Vol:", thermodynamic_properties.volume)
    print("Post_Box_len:", np.power(thermodynamic_properties.volume, 1 / 3))
    print("End Time:",time.ctime())
    print("--- End Of Simulation ---")

#Extend MD Simulation
def extended_brownian_dynamics_simulation(snapshot_file, kT, Viscosity, EnergyScaling, dt, ext_trigger_rate, seed, time, LJCut,Extended_time, kappa):
    # Open the snapshot file
    gsd.hoomd.open(name=snapshot_file, mode='r')

    # Define Sim parametersslow_trigger_rate
    gpu = hoomd.device.GPU()
    sim = hoomd.Simulation(device=gpu, seed=seed)
    sim.create_state_from_gsd(filename=snapshot_file)

    # Define LJ interactions between particles
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4))

    # Define Yukawa interactions between particles
    yukawa = hoomd.md.pair.Yukawa(nlist=hoomd.md.nlist.Cell(buffer=0.4))

    for i, species_vec_1 in enumerate(SpeciesList):
        for j, species_vec_2 in enumerate(SpeciesList):


            species1_name = species_vec_1[0]
            species2_name = species_vec_2[0]
            radius1 = species_vec_1[4]
            radius2 = species_vec_2[4]
            charge1 = species_vec_1[1]
            charge2 = species_vec_2[1]
            epsilon = InteractionList[i][j]
            sigma = (radius1 + radius2) / 2
            lj.params[(species1_name, species2_name)] = dict(epsilon=epsilon*EnergyScaling, sigma=sigma)

            if  LJCut == True:
                lj.r_cut[(species1_name, species2_name)] = (2**(1/6))*sigma
            else:
                lj.r_cut[(species1_name, species2_name)] = (3*sigma)

            # Set Yukawa parameters (You might need to adjust these values according to your needs)
            kappa =  kappa # Calculated Kappa as inverse Debye Length
            epsilon_yukawa = charge1 * charge2   # Scale Yukawa epsilon based on coulombs law and EnergyScaling

            yukawa.params[(species1_name, species2_name)] = dict(epsilon=epsilon_yukawa, kappa=kappa)

            # Setting cut-off radius for Yukawa (Adjust this value according to your needs)
            yukawa.r_cut[(species1_name, species2_name)] = lj.r_cut[(species1_name, species2_name)]

    # Brownian Dynamics
    brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=kT, default_gamma=1.0, default_gamma_r=(1.0, 1.0, 1.0))

    # Set custom gamma values for each species based on species_list
    for species_info in SpeciesList:
        species_name = species_info[0]
        species_radius = species_info[4]
        gamma = (6 * np.pi * Viscosity * species_radius)
        brownian.gamma[species_name] = gamma

    # Integrator
    integrator = hoomd.md.Integrator(dt=dt, methods=[brownian])

    # Add LJ potential to the integrator
    integrator.forces.append(lj)

    # Add LJ and Yukawa potentials to the integrator
    integrator.forces.extend([lj, yukawa])

    # Add integrator to the simulation
    sim.operations.integrator = integrator

    # Set random velocities and equilibrate
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

    # Monitoring Properties
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermodynamic_properties)

    sim.run(0)

    sim.run(1)
    print("--- New Simulation ---")
    print("Start Time:",time.ctime())
    print("Seed:",seed)
    print("kT:",kT)
    print("EnergyScaling:",EnergyScaling)
    print("Eq_DOF:", thermodynamic_properties.degrees_of_freedom)
    print("Eq_KE:", thermodynamic_properties.kinetic_energy)
    print("Eq_PE:", thermodynamic_properties.potential_energy)
    print("Eq_Kin_Temp:", thermodynamic_properties.kinetic_temperature)
    print("Eq_Vol:", thermodynamic_properties.volume)
    print("Given BoxLength", box_length_final)
    print("Eq_Box_len:", np.power(thermodynamic_properties.volume, 1 / 3))

    logger = hoomd.logging.Logger()
    logger.add(thermodynamic_properties)
    logger.add(sim, quantities=['timestep', 'walltime'])

    # Create a slow writer with dynamic property and momentum
    gsd_writer = hoomd.write.GSD(
        filename='trajectory.gsd',
        trigger=hoomd.trigger.Periodic(slow_trigger_rate),
        mode='wb',  # 'wb' to start a new file
        dynamic=['property', 'momentum']
    )
    gsd_writer.logger = logger  # Attach the logger to the writer

    # Add the writer to the simulation
    sim.operations.writers.append(gsd_writer)

    # Run the simulation with the slow writer for switch_step steps
    sim.run(switch_step)

    # Remove the slow writer and add the fast writer
    sim.operations.writers.remove(gsd_writer)
    gsd_writer = hoomd.write.GSD(
        filename='trajectory.gsd',
        trigger=hoomd.trigger.Periodic(fast_trigger_rate),
        mode='ab',  # 'ab' to append to the existing file
        dynamic=['property', 'momentum']
    )
    gsd_writer.logger = logger  # Attach the logger to the writer
    sim.operations.writers.append(gsd_writer)

    # Continue running the simulation with the fast writer
    sim.run(TotalTime - switch_step)  # Run for the remaining steps

    print("--- Running Extended Simulation ---")
    print("Start Time:",time.ctime())
    print("Seed:",seed)
    print("kT:",kT)
    print("EnergyScaling:",EnergyScaling)
    print("Eq_DOF:", thermodynamic_properties.degrees_of_freedom)
    print("Eq_KE:", thermodynamic_properties.kinetic_energy)
    print("Eq_PE:", thermodynamic_properties.potential_energy)
    print("Eq_Kin_Temp:", thermodynamic_properties.kinetic_temperature)
    print("Eq_Vol:", thermodynamic_properties.volume)
    print("Given BoxLength", box_length_final)
    print("Eq_Box_len:", np.power(thermodynamic_properties.volume, 1 / 3))

    sim.operations.writers.remove(gsd_writer)
    gsd_writer = hoomd.write.GSD(
        filename='trajectory.gsd',
        trigger=hoomd.trigger.Periodic(ext_trigger_rate),
        mode='ab',  # 'ab' to append to the existing file
        dynamic=['property', 'momentum']
    )
    gsd_writer.logger = logger  # Attach the logger to the writer
    sim.operations.writers.append(gsd_writer)

    # Continue running the simulation with the fast writer
    sim.run(Extended_time - TotalTime + switch_step)  # Run for the remaining steps

    print("--- Done Running Extended Simulation ---")
    print("Post_DOF:", thermodynamic_properties.degrees_of_freedom)
    print("Post_KE:", thermodynamic_properties.kinetic_energy)
    print("Post_PE:", thermodynamic_properties.potential_energy)
    print("Post_Kin_Temp:", thermodynamic_properties.kinetic_temperature)
    print("Post_Vol:", thermodynamic_properties.volume)
    print("Post_Box_len:", np.power(thermodynamic_properties.volume, 1 / 3))
    print("End Time:",time.ctime())
    print("--- End Of Extended Simulation ---")

#MSD 2 traj
def MSDanalysis2(snapshot_file1, snapshot_file2, dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor):
    print('--------------------')
    print('Analyzing Trajectories')

    # Open the trajectory files
    traj1 = gsd.hoomd.open(snapshot_file1, 'r')
    traj2 = gsd.hoomd.open(snapshot_file2, 'r')

    data1 = gsd.hoomd.read_log(snapshot_file1)
    data2 = gsd.hoomd.read_log(snapshot_file2)

    D_table = []

    print(list(data1.keys()))
    print(list(data2.keys()))

    # Extract the quantities of interest from the data dictionaries
    timestep1 = data1['log/Simulation/timestep']
    timestep2 = data2['log/Simulation/timestep']

    # Create MSD objects for each trajectory
    box1 = traj1[0].configuration.box
    box2 = traj2[0].configuration.box
    msd1 = freud.msd.MSD(mode='direct', box=box1)
    msd2 = freud.msd.MSD(mode='direct', box=box2)

    # Initialize a list to store CSV data
    csv_data = []

    # Iterate over each Species2Monitor
    for Species2Monitor in Species2Monitor:
        print(f'--------------------')
        print(f'Analyzing Species: {Species2Monitor}')

        # Create four subplots for the current Species2Monitor
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        # Iterate over each species
        for i, species in enumerate(SpeciesList):
            species_name = species[0]

            if species_name != Species2Monitor:
                continue  # Skip species that don't match Species2Monitor

            print('Monitoring this species:')
            print(Species2Monitor)

            radius = species[4]  # Extract radius from SpeciesList

            # Calculate D_exp using Stokes-Einstein relation
            D_exp = (kT) / (Viscosity * np.pi * radius)

            # Collect positions for particles of the current species from both trajectories
            positions_species = []
            pos1 = []
            pos2 = []
            image1 = []
            image2 = []

            # Iterate over each frame in the first trajectory
            for frame in traj1:
                positions = frame.particles.position
                types = frame.particles.typeid
                species_indices = np.where(types == i)[0]  # Filter particles of species i
                positions = frame.particles.position[species_indices]
                image1.append(frame.particles.image[species_indices])
                pos1.append(positions)

            # Iterate over each frame in the second trajectory
            for frame in traj2:
                positions = frame.particles.position
                types = frame.particles.typeid
                species_indices = np.where(types == i)[0]  # Filter particles of species i
                positions = frame.particles.position[species_indices]
                image2.append(frame.particles.image[species_indices])
                pos2.append(positions)

            # Convert positions to numpy arrays
            positions_species1 = np.array(pos1)
            positions_species2 = np.array(pos2)
            image_species1 = np.array(image1)
            image_species2 = np.array(image2)

            # Concatenate the positions and images from both trajectories
            positions_species = np.concatenate((positions_species1, positions_species2))
            image_species = np.concatenate((image_species1, image_species2))

            # Access MSD data for both trajectories
            avg_msd_data1 = msd1.compute(positions_species1, images=image_species1, reset=True)
            avg_msd_data2 = msd2.compute(positions_species2, images=image_species2, reset=True)

            # Calculate MSD values for both trajectories
            msd_values1 = avg_msd_data1.msd
            msd_values2 = avg_msd_data2.msd

            # Concatenate MSD values from both trajectories
            msd_values = np.concatenate((msd_values1, msd_values2))

            # Calculate time intervals for gradients
            time_points1 = timestep1 * dt
            time_points1 = time_points1 - time_points1[0]
            time_points2 = timestep2 * dt
            time_points2 = time_points2 - time_points2[0]
            time_points = np.concatenate((time_points1, time_points2))

            time_points_forlog = np.arange(len(msd_values)) * Traj_Write_Rate

            # Calculate gradients of MSD vs. Time and log(MSD) vs. log(Time) using central difference
            msd_gradient = np.gradient(msd_values, time_points)
            log_msd_values = np.log(msd_values)
            log_time_points = np.log(time_points_forlog)
            log_msd_log_gradient = np.gradient(log_msd_values, log_time_points)

            # Additional plots for each subplot
            ax1.plot(time_points, msd_values, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            for particle_idx in range(len(species_indices)):
                particle_msd_values = avg_msd_data1.particle_msd[:, particle_idx]
                ax1.plot(time_points1, particle_msd_values, '-', color=f'C{i}', alpha=0.1)
            ax1.plot(time_points, D_exp * (time_points), '--', color='black', label=f'{species_name} - D_exp')
            ax1.set_xlabel('Time [ps]')
            ax1.set_ylabel('MSD')
            ax1.set_title(f'Linear MSD Analysis: {Species2Monitor} - {params_name}')
            ax1.legend()

            ax2.plot(time_points, msd_values, 'o', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax2.plot(time_points, D_exp * (time_points), '--', color='black', label=f'{species_name} - D_exp')
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel('Log Time [ps]')
            ax2.set_ylabel('Log MSD')
            ax2.set_title(f'Log-Log MSD Analysis: {Species2Monitor} - {params_name}')
            ax2.legend()

            ax3.plot(time_points, msd_gradient, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax3.axhline(y=D_exp, color='black', linestyle='--', label='D_exp')
            ax3.set_xlabel('Time [ps]')
            ax3.set_ylabel('Gradient of MSD')
            ax3.set_title(f'Gradient of MSD vs. Time: {Species2Monitor} - {params_name}')
            ax3.legend()

            ax4.plot(time_points, log_msd_log_gradient, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax4.axhline(y=1, color='black', linestyle='--', label='Alpha_exp=1')
            ax4.set_xlabel('Time [ps]')
            ax4.set_ylabel('Gradient of Log MSD')
            ax4.set_title(f'Gradient of Log MSD vs. Log Time: {Species2Monitor} - {params_name}')
            ax4.legend()

            plt.tight_layout()
            plt.savefig(f'MSD_{Species2Monitor}_{params_name}.png')
            plt.close()

            # Calculate time average for D_fit and alpha_fit
            D_fit_values = msd_gradient[~np.isnan(msd_gradient)]
            alpha_fit_values = log_msd_log_gradient[~np.isnan(log_msd_log_gradient)]
            D_fit = np.mean(D_fit_values) if len(D_fit_values) > 0 else np.nan
            alpha_fit = np.mean(alpha_fit_values) if len(alpha_fit_values) > 0 else np.nan

            # Organize data for CSV
            species_csv_data = {
                'Species': species_name,
                'Time [ps]': time_points.tolist(),
                'MSD Values': msd_values.tolist(),
                'MSD Gradient': msd_gradient.tolist(),
                'Log MSD Log Gradient': log_msd_log_gradient.tolist()
            }
            # Append the species data to the csv_data list
            csv_data.append(species_csv_data)

        print(f'End of MSD Analysis for Species: {Species2Monitor}')
        print(f'--------------------')

    # Create a table using numpy
    # Debugging: Check the consistency of D_table
    # Check if D_table is empty
    if not D_table:
        print("D_table is empty. Skipping table creation.")
    else:
        try:
            # Create a table using numpy
            D_table = np.array(D_table)
            column_names = ['Species', 'D_exp', 'D_fit', 'alpha_fit']
            table = np.vstack((column_names, D_table))

            np.savetxt(f'Table_{params_name}.csv', table, delimiter=',', fmt='%s')

            # Plot the table as a PNG image
            fig_table, ax_table = plt.subplots(figsize=(8, 6))
            ax_table.axis('off')
            table_data = []
            for row in table:
                table_data.append(row)
            table_img = ax_table.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', cellColours=None)
            table_img.auto_set_font_size(False)
            table_img.set_fontsize(12)
            table_img.scale(1.2, 1.2)
            plt.savefig(f'Table_{params_name}.png', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        except ValueError as e:
            print(f"Error encountered while creating D_table: {e}")
            print("Skipping D_table creation.")

    # After all species have been processed
    for species_data in csv_data:
        species_name = species_data['Species']
        df = pd.DataFrame({
            'Time [ps]': species_data['Time [ps]'],
            'MSD Values': species_data['MSD Values'],
            'MSD Gradient': species_data['MSD Gradient'],
            'Log MSD Log Gradient': species_data['Log MSD Log Gradient']
        })
        df.to_csv(f'MSD_{species_name}_{params_name}.csv', index=False)

    print('CSV files saved for each species.')
    print('End of all MSD Analysis')
    print('--------------------')

#MSD 1 traj
def MSDanalysis1(snapshot_file1, dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor):
    print('--------------------')
    print('Analyzing Trajectories')

    # Open the trajectory files
    traj1 = gsd.hoomd.open(snapshot_file1, 'r')

    data1 = gsd.hoomd.read_log(snapshot_file1)

    D_table = []

    print(list(data1.keys()))

    # Extract the quantities of interest from the data dictionaries
    timestep1 = data1['log/Simulation/timestep']

    # Create MSD objects for each trajectory
    box1 = traj1[0].configuration.box
    msd1 = freud.msd.MSD(mode='direct', box=box1)

    # Initialize a list to store CSV data
    csv_data = []

    # Iterate over each Species2Monitor
    for Species2Monitor in Species2Monitor:
        print(f'--------------------')
        print(f'Analyzing Species: {Species2Monitor}')

        # Create four subplots for the current Species2Monitor
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        # Iterate over each species
        for i, species in enumerate(SpeciesList):
            species_name = species[0]

            if species_name != Species2Monitor:
                continue  # Skip species that don't match Species2Monitor

            print('Monitoring this species:')
            print(Species2Monitor)

            radius = species[4]  # Extract radius from SpeciesList

            # Calculate D_exp using Stokes-Einstein relation
            D_exp = (kT) / (Viscosity * np.pi * radius)

            # Collect positions for particles of the current species from both trajectories
            positions_species = []
            pos1 = []
            image1 = []

            # Iterate over each frame in the first trajectory
            for frame in traj1:
                positions = frame.particles.position
                types = frame.particles.typeid
                species_indices = np.where(types == i)[0]  # Filter particles of species i
                positions = frame.particles.position[species_indices]
                image1.append(frame.particles.image[species_indices])
                pos1.append(positions)

            # Convert positions to numpy arrays
            positions_species1 = np.array(pos1)
            image_species1 = np.array(image1)

            # Access MSD data for the trajectory
            avg_msd_data1 = msd1.compute(positions_species1, images=image_species1, reset=True)

            # Calculate MSD values for the trajectory
            msd_values1 = avg_msd_data1.msd

            # Calculate time intervals for gradients
            time_points1 = timestep1 * dt
            time_points1 = time_points1 - time_points1[0]
            time_points = time_points1

            time_points_forlog = np.arange(len(msd_values1)) * Traj_Write_Rate

            # Calculate gradients of MSD vs. Time and log(MSD) vs. log(Time) using central difference
            msd_gradient = np.gradient(msd_values1, time_points)
            log_msd_values = np.log(msd_values1)
            log_time_points = np.log(time_points_forlog)
            log_msd_log_gradient = np.gradient(log_msd_values, log_time_points)

            # Additional plots for each subplot
            ax1.plot(time_points, msd_values1, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax1.plot(time_points, D_exp * (time_points), '--', color='black', label=f'{species_name} - D_exp')
            ax1.set_xlabel('Time [ps]')
            ax1.set_ylabel('MSD')
            ax1.set_title(f'Linear MSD Analysis: {Species2Monitor} - {params_name}')
            ax1.legend()

            ax2.plot(time_points, msd_values1, 'o', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax2.plot(time_points, D_exp * (time_points), '--', color='black', label=f'{species_name} - D_exp')
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel('Log Time [ps]')
            ax2.set_ylabel('Log MSD')
            ax2.set_title(f'Log-Log MSD Analysis: {Species2Monitor} - {params_name}')
            ax2.legend()

            ax3.plot(time_points, msd_gradient, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax3.axhline(y=D_exp, color='black', linestyle='--', label='D_exp')
            ax3.set_xlabel('Time [ps]')
            ax3.set_ylabel('Gradient of MSD')
            ax3.set_title(f'Gradient of MSD vs. Time: {Species2Monitor} - {params_name}')
            ax3.legend()

            ax4.plot(time_points, log_msd_log_gradient, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax4.axhline(y=1, color='black', linestyle='--', label='Alpha_exp=1')
            ax4.set_xlabel('Time [ps]')
            ax4.set_ylabel('Gradient of Log MSD')
            ax4.set_title(f'Gradient of Log MSD vs. Log Time: {Species2Monitor} - {params_name}')
            ax4.legend()

            plt.tight_layout()
            plt.savefig(f'MSD_{Species2Monitor}_{params_name}.png')
            plt.close()

            # Calculate time average for D_fit and alpha_fit
            D_fit_values = msd_gradient[~np.isnan(msd_gradient)]
            alpha_fit_values = log_msd_log_gradient[~np.isnan(log_msd_log_gradient)]
            D_fit = np.mean(D_fit_values) if len(D_fit_values) > 0 else np.nan
            alpha_fit = np.mean(alpha_fit_values) if len(alpha_fit_values) > 0 else np.nan

            # Organize data for CSV
            species_csv_data = {
                'Species': species_name,
                'Time [ps]': time_points.tolist(),
                'MSD Values': msd_values1.tolist(),
                'MSD Gradient': msd_gradient.tolist(),
                'Log MSD Log Gradient': log_msd_log_gradient.tolist()
            }
            # Append the species data to the csv_data list
            csv_data.append(species_csv_data)

        print(f'End of MSD Analysis for Species: {Species2Monitor}')
        print(f'--------------------')

    # Debugging: Check the consistency of D_table
    # Check if D_table is empty
    if not D_table:
        print("D_table is empty. Skipping table creation.")
    else:
        try:
            # Create a table using numpy
            D_table = np.array(D_table)
            column_names = ['Species', 'D_exp', 'D_fit', 'alpha_fit']
            table = np.vstack((column_names, D_table))

            np.savetxt(f'Table_{params_name}.csv', table, delimiter=',', fmt='%s')

            # Plot the table as a PNG image
            fig_table, ax_table = plt.subplots(figsize=(8, 6))
            ax_table.axis('off')
            table_data = []
            for row in table:
                table_data.append(row)
            table_img = ax_table.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', cellColours=None)
            table_img.auto_set_font_size(False)
            table_img.set_fontsize(12)
            table_img.scale(1.2, 1.2)
            plt.savefig(f'Table_{params_name}.png', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        except ValueError as e:
            print(f"Error encountered while creating D_table: {e}")
            print("Skipping D_table creation.")

    # After all species have been processed
    for species_data in csv_data:
        species_name = species_data['Species']
        df = pd.DataFrame({
            'Time [ps]': species_data['Time [ps]'],
            'MSD Values': species_data['MSD Values'],
            'MSD Gradient': species_data['MSD Gradient'],
            'Log MSD Log Gradient': species_data['Log MSD Log Gradient']
        })
        df.to_csv(f'MSD_{species_name}_{params_name}.csv', index=False)

    print('CSV files saved for each species.')
    print('End of all MSD Analysis')
    print('--------------------')

#MSD ramp traj
def MSDanalysisRamp(snapshot_file1, dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor):
    print('--------------------')
    print('Analyzing Trajectories')

    # Open the trajectory files
    traj1 = gsd.hoomd.open(snapshot_file1, 'r')

    data1 = gsd.hoomd.read_log(snapshot_file1)

    D_table = []

    print(list(data1.keys()))

    # Extract the quantities of interest from the data dictionaries
    timestep1 = data1['log/Simulation/timestep']

    # Create MSD objects for each trajectory
    box1 = traj1[0].configuration.box
    msd1 = freud.msd.MSD(mode='direct', box=box1)

    # Initialize a list to store CSV data
    csv_data = []

    # Iterate over each Species2Monitor
    for Species2Monitor in Species2Monitor:
        print(f'--------------------')
        print(f'Analyzing Species: {Species2Monitor}')

        # Create four subplots for the current Species2Monitor
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        # Iterate over each species
        for i, species in enumerate(SpeciesList):
            species_name = species[0]

            if species_name != Species2Monitor:
                continue  # Skip species that don't match Species2Monitor

            print('Monitoring this species:')
            print(Species2Monitor)

            radius = species[4]  # Extract radius from SpeciesList

            # Calculate D_exp using Stokes-Einstein relation
            D_exp = (kT) / (Viscosity * np.pi * radius)

            # Collect positions for particles of the current species from both trajectories
            positions_species = []
            pos1 = []
            image1 = []

            # Iterate over each frame in the first trajectory
            for frame in traj1:
                positions = frame.particles.position
                types = frame.particles.typeid
                species_indices = np.where(types == i)[0]  # Filter particles of species i
                positions = frame.particles.position[species_indices]
                image1.append(frame.particles.image[species_indices])
                pos1.append(positions)

            # Convert positions to numpy arrays
            positions_species1 = np.array(pos1)
            image_species1 = np.array(image1)

            # Access MSD data for the trajectory
            avg_msd_data1 = msd1.compute(positions_species1, images=image_species1, reset=True)

            # Calculate MSD values for the trajectory
            msd_values1 = avg_msd_data1.msd

            # Calculate time intervals for gradients
            time_points1 = timestep1 * dt
            time_points1 = time_points1 - time_points1[0]
            time_points = time_points1

            time_points_forlog = np.arange(len(msd_values1)) * Traj_Write_Rate

            # Calculate gradients of MSD vs. Time and log(MSD) vs. log(Time) using central difference
            msd_gradient = np.gradient(msd_values1, time_points)
            log_msd_values = np.log(msd_values1)
            log_time_points = np.log(time_points_forlog)
            log_msd_log_gradient = np.gradient(log_msd_values, log_time_points)

            # Additional plots for each subplot
            ax1.plot(time_points, msd_values1, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax1.plot(time_points, D_exp * (time_points), '--', color='black', label=f'{species_name} - D_exp')
            ax1.set_xlabel('Time [ps]')
            ax1.set_ylabel('MSD')
            ax1.set_title(f'Linear MSD Analysis: {Species2Monitor} - {params_name}')
            ax1.legend()

            ax2.plot(time_points, msd_values1, 'o', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax2.plot(time_points, D_exp * (time_points), '--', color='black', label=f'{species_name} - D_exp')
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel('Log Time [ps]')
            ax2.set_ylabel('Log MSD')
            ax2.set_title(f'Log-Log MSD Analysis: {Species2Monitor} - {params_name}')
            ax2.legend()

            ax3.plot(time_points, msd_gradient, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax3.axhline(y=D_exp, color='black', linestyle='--', label='D_exp')
            ax3.set_xlabel('Time [ps]')
            ax3.set_ylabel('Gradient of MSD')
            ax3.set_title(f'Gradient of MSD vs. Time: {Species2Monitor} - {params_name}')
            ax3.legend()

            ax4.plot(time_points, log_msd_log_gradient, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax4.axhline(y=1, color='black', linestyle='--', label='Alpha_exp=1')
            ax4.set_xlabel('Time [ps]')
            ax4.set_ylabel('Gradient of Log MSD')
            ax4.set_title(f'Gradient of Log MSD vs. Log Time: {Species2Monitor} - {params_name}')
            ax4.legend()

            plt.tight_layout()
            plt.savefig(f'MSD_Ramp_{Species2Monitor}_{params_name}.png')
            plt.close()

            # Calculate time average for D_fit and alpha_fit
            D_fit_values = msd_gradient[~np.isnan(msd_gradient)]
            alpha_fit_values = log_msd_log_gradient[~np.isnan(log_msd_log_gradient)]
            D_fit = np.mean(D_fit_values) if len(D_fit_values) > 0 else np.nan
            alpha_fit = np.mean(alpha_fit_values) if len(alpha_fit_values) > 0 else np.nan

            # Organize data for CSV
            species_csv_data = {
                'Species': species_name,
                'Time [ps]': time_points.tolist(),
                'MSD Values': msd_values1.tolist(),
                'MSD Gradient': msd_gradient.tolist(),
                'Log MSD Log Gradient': log_msd_log_gradient.tolist()
            }
            # Append the species data to the csv_data list
            csv_data.append(species_csv_data)

        print(f'End of MSD Analysis for Species: {Species2Monitor}')
        print(f'--------------------')

    # Debugging: Check the consistency of D_table
    # Check if D_table is empty
    if not D_table:
        print("D_table is empty. Skipping table creation.")
    else:
        try:
            # Create a table using numpy
            D_table = np.array(D_table)
            column_names = ['Species', 'D_exp', 'D_fit', 'alpha_fit']
            table = np.vstack((column_names, D_table))

            np.savetxt(f'Table_{params_name}.csv', table, delimiter=',', fmt='%s')

            # Plot the table as a PNG image
            fig_table, ax_table = plt.subplots(figsize=(8, 6))
            ax_table.axis('off')
            table_data = []
            for row in table:
                table_data.append(row)
            table_img = ax_table.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', cellColours=None)
            table_img.auto_set_font_size(False)
            table_img.set_fontsize(12)
            table_img.scale(1.2, 1.2)
            plt.savefig(f'Table_Ramp_{params_name}.png', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        except ValueError as e:
            print(f"Error encountered while creating D_table: {e}")
            print("Skipping D_table creation.")

    # After all species have been processed
    for species_data in csv_data:
        species_name = species_data['Species']
        df = pd.DataFrame({
            'Time [ps]': species_data['Time [ps]'],
            'MSD Values': species_data['MSD Values'],
            'MSD Gradient': species_data['MSD Gradient'],
            'Log MSD Log Gradient': species_data['Log MSD Log Gradient']
        })
        df.to_csv(f'MSD_Ramp_{species_name}_{params_name}.csv', index=False)

    print('CSV files saved for each species.')
    print('End of all MSD Analysis')
    print('--------------------')

#MSD 3 traj
def MSDanalysis3(snapshot_file1, dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor):
    print('--------------------')
    print('Analyzing Trajectories')

    # Open the trajectory files
    traj1 = gsd.hoomd.open(snapshot_file1, 'r')

    data1 = gsd.hoomd.read_log(snapshot_file1)

    D_table = []

    print(list(data1.keys()))

    # Extract the quantities of interest from the data dictionaries
    timestep1 = data1['log/Simulation/timestep']

    # Create MSD objects for each trajectory
    box1 = traj1[0].configuration.box
    msd1 = freud.msd.MSD(mode='direct', box=box1)

    # Initialize a list to store CSV data
    csv_data = []

    # Iterate over each Species2Monitor
    for Species2Monitor in Species2Monitor:
        print(f'--------------------')
        print(f'Analyzing Species: {Species2Monitor}')

        # Create four subplots for the current Species2Monitor
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        # Iterate over each species
        for i, species in enumerate(SpeciesList):
            species_name = species[0]

            if species_name != Species2Monitor:
                continue  # Skip species that don't match Species2Monitor

            print('Monitoring this species:')
            print(Species2Monitor)

            radius = species[4]  # Extract radius from SpeciesList

            # Calculate D_exp using Stokes-Einstein relation
            D_exp = (kT) / (Viscosity * np.pi * radius)

            # Collect positions for particles of the current species from both trajectories
            positions_species = []
            pos1 = []
            image1 = []

            # Iterate over each frame in the first trajectory
            for frame in traj1:
                positions = frame.particles.position
                types = frame.particles.typeid
                species_indices = np.where(types == i)[0]  # Filter particles of species i
                positions = frame.particles.position[species_indices]
                image1.append(frame.particles.image[species_indices])
                pos1.append(positions)

            # Convert positions to numpy arrays
            positions_species1 = np.array(pos1)
            image_species1 = np.array(image1)

            # Access MSD data for the trajectory
            avg_msd_data1 = msd1.compute(positions_species1, images=image_species1, reset=True)

            # Calculate MSD values for the trajectory
            msd_values1 = avg_msd_data1.msd

            # Calculate time intervals for gradients
            time_points1 = timestep1 * dt
            time_points1 = time_points1 - time_points1[0]
            time_points = time_points1

            time_points_forlog = np.arange(len(msd_values1)) * Traj_Write_Rate

            # Calculate gradients of MSD vs. Time and log(MSD) vs. log(Time) using central difference
            msd_gradient = np.gradient(msd_values1, time_points)
            log_msd_values = np.log(msd_values1)
            log_time_points = np.log(time_points_forlog)
            log_msd_log_gradient = np.gradient(log_msd_values, log_time_points)

            # Additional plots for each subplot
            ax1.plot(time_points, msd_values1, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax1.plot(time_points, D_exp * (time_points), '--', color='black', label=f'{species_name} - D_exp')
            ax1.set_xlabel('Time [ps]')
            ax1.set_ylabel('MSD')
            ax1.set_title(f'Linear MSD Analysis: {Species2Monitor} - {params_name}')
            ax1.legend()

            ax2.plot(time_points, msd_values1, 'o', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax2.plot(time_points, D_exp * (time_points), '--', color='black', label=f'{species_name} - D_exp')
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel('Log Time [ps]')
            ax2.set_ylabel('Log MSD')
            ax2.set_title(f'Log-Log MSD Analysis: {Species2Monitor} - {params_name}')
            ax2.legend()

            ax3.plot(time_points, msd_gradient, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax3.axhline(y=D_exp, color='black', linestyle='--', label='D_exp')
            ax3.set_xlabel('Time [ps]')
            ax3.set_ylabel('Gradient of MSD')
            ax3.set_title(f'Gradient of MSD vs. Time: {Species2Monitor} - {params_name}')
            ax3.legend()

            ax4.plot(time_points, log_msd_log_gradient, 'o-', color=f'C{i}', label=f'{species_name} (Avg)', markersize=5)
            ax4.axhline(y=1, color='black', linestyle='--', label='Alpha_exp=1')
            ax4.set_xlabel('Time [ps]')
            ax4.set_ylabel('Gradient of Log MSD')
            ax4.set_title(f'Gradient of Log MSD vs. Log Time: {Species2Monitor} - {params_name}')
            ax4.legend()

            plt.tight_layout()
            plt.savefig(f'MSD_{Species2Monitor}_{params_name}.png')
            plt.close()

            # Calculate time average for D_fit and alpha_fit
            D_fit_values = msd_gradient[~np.isnan(msd_gradient)]
            alpha_fit_values = log_msd_log_gradient[~np.isnan(log_msd_log_gradient)]
            D_fit = np.mean(D_fit_values) if len(D_fit_values) > 0 else np.nan
            alpha_fit = np.mean(alpha_fit_values) if len(alpha_fit_values) > 0 else np.nan

            # Organize data for CSV
            species_csv_data = {
                'Species': species_name,
                'Time [ps]': time_points.tolist(),
                'MSD Values': msd_values1.tolist(),
                'MSD Gradient': msd_gradient.tolist(),
                'Log MSD Log Gradient': log_msd_log_gradient.tolist()
            }
            # Append the species data to the csv_data list
            csv_data.append(species_csv_data)

        print(f'End of MSD Analysis for Species: {Species2Monitor}')
        print(f'--------------------')

    # Debugging: Check the consistency of D_table
    # Check if D_table is empty
    if not D_table:
        print("D_table is empty. Skipping table creation.")
    else:
        try:
            # Create a table using numpy
            D_table = np.array(D_table)
            column_names = ['Species', 'D_exp', 'D_fit', 'alpha_fit']
            table = np.vstack((column_names, D_table))

            np.savetxt(f'Table_{params_name}.csv', table, delimiter=',', fmt='%s')

            # Plot the table as a PNG image
            fig_table, ax_table = plt.subplots(figsize=(8, 6))
            ax_table.axis('off')
            table_data = []
            for row in table:
                table_data.append(row)
            table_img = ax_table.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', cellColours=None)
            table_img.auto_set_font_size(False)
            table_img.set_fontsize(12)
            table_img.scale(1.2, 1.2)
            plt.savefig(f'Table_{params_name}.png', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        except ValueError as e:
            print(f"Error encountered while creating D_table: {e}")
            print("Skipping D_table creation.")

    # After all species have been processed
    for species_data in csv_data:
        species_name = species_data['Species']
        df = pd.DataFrame({
            'Time [ps]': species_data['Time [ps]'],
            'MSD Values': species_data['MSD Values'],
            'MSD Gradient': species_data['MSD Gradient'],
            'Log MSD Log Gradient': species_data['Log MSD Log Gradient']
        })
        df.to_csv(f'MSD_Ramp_{species_name}_{params_name}.csv', index=False)

    print('CSV files saved for each species.')
    print('End of all MSD Analysis')
    print('--------------------')

#RDF using Freud
def RDFAnalysis(snapshot_file, species_list, box_length):
    try:
        # Open the trajectory file
        traj = gsd.hoomd.open(snapshot_file, 'r')
        box = traj[0].configuration.box

        # Initialize the RDF calculator
        rdf = freud.density.RDF(bins=100, r_max=box_length / 2)

        # Create a mapping of species names to type IDs
        type_mapping = {name: index for index, name in enumerate(traj[0].particles.types)}

        for frame in traj:
            step = frame.configuration.step
            positions = frame.particles.position
            types = frame.particles.typeid
            total_particles = len(positions)

            for species_to_monitor in species_list:
                species_id = type_mapping.get(species_to_monitor, None)

                # Check if the species name is valid
                if species_id is None:
                    print(f'Species {species_to_monitor} not found in frame {step}')
                    continue

                species_indices = np.where(types == species_id)[0]
                positions_species = positions[species_indices]

                rdf_values = []

                for other_species in species_list:
                    if other_species == species_to_monitor:
                        continue

                    other_species_id = type_mapping.get(other_species, None)

                    if other_species_id is None:
                        print(f'Species {other_species} not found in frame {step}')
                        continue

                    other_species_indices = np.where(types == other_species_id)[0]
                    positions_other_species = positions[other_species_indices]

                    rdf.compute(system=(box, positions_species), query_points=positions_other_species, reset=True)
                    rdf_values_normalized = rdf.rdf / (total_particles - len(species_indices))  # Normalize by the number of other particles

                    rdf_values.append(rdf_values_normalized)

                # Plot RDF for this species against all other species
                if rdf_values:
                    print(f'Plotting RDF for Species {species_to_monitor} - Frame {step}')
                    plt.figure(figsize=(10, 5))
                    for i, other_species in enumerate(species_list):
                        if other_species != species_to_monitor:
                            plt.plot(rdf.bin_centers, rdf_values[i], label=f'{other_species} - {step}')

                    plt.xlabel('Distance [Normalized]')
                    plt.ylabel('Normalized RDF')
                    plt.title(f'RDF for Species {species_to_monitor} - Frame {step}')
                    plt.xlim(0, 1)  # X-axis normalized to box size
                    plt.ylim(0, 1)  # Y-axis normalized to total number of other particles
                    plt.legend()
                    plt.grid(True)

                    # Save the plot with a unique filename
                    plot_filename = f'RDF_{species_to_monitor}_Frame_{step}.png'
                    plt.savefig(plot_filename)
                    plt.close()
                    print(f'Saved plot as: {plot_filename}')

    except Exception as e:
        print(f"Error in RDFAnalysis: {str(e)}")

#RDF using Freud
def OldRDFAnalysis(snapshot_file, params_name, species_list):
    # Open the trajectory file
    traj = gsd.hoomd.open(snapshot_file, 'r')
    box_length_mb = box_length/10
    # Get the box dimensions
    box_dimensions = np.array([box_length_mb, box_length_mb, box_length_mb])

    # Create RDF object
    rdf = freud.density.RDF(bins=1000, r_max=(box_length_mb/20), normalize=True)

    # Iterate over each species pair
    for i, species1 in enumerate(species_list):
        species1_name = species1[0]

        for j, species2 in enumerate(species_list):
            species2_name = species2[0]

            # Collect positions for particles of the current species pair
            positions_species_pair = []

            # Iterate over each frame
            for frame in traj:
                positions = frame.particles.position
                types = frame.particles.typeid
                species1_indices = np.where(types == i)[0]  # Filter particles of species i
                species2_indices = np.where(types == j)[0]  # Filter particles of species j

                # Create box object
                box = freud.box.Box.from_box(box_dimensions)

                # Compute RDF for the species pair
                rdf.compute(system=(box, positions[species1_indices]))

            # Plot RDF for the species pair
            plt.plot(rdf.bin_centers, rdf.rdf, label=f'{species1_name}-{species2_name}')

    # Set plot title and labels
    plt.xlabel('Distance')
    plt.ylabel('Radial Distribution Function')
    plt.title(f'Radial Distribution Function Analysis: {params_name}')
    plt.legend()
    plt.savefig(f'RDF_{params_name}.png')
    plt.close()

# Monitoring 1 Traj
def monitor1(snapshot_file, kT, params_name, dt):
    # Read the logged data from the GSD file
    data = gsd.hoomd.read_log(snapshot_file)

    # Extract the quantities of interest from the data dictionary
    timestep = data['configuration/step']
    potential_energy = data['log/md/compute/ThermodynamicQuantities/potential_energy']
    kinetic_temperature = data['log/md/compute/ThermodynamicQuantities/kinetic_temperature']
    pressure = data['log/md/compute/ThermodynamicQuantities/pressure']
    kinetic_energy = data['log/md/compute/ThermodynamicQuantities/kinetic_energy']
    total_energy = potential_energy + kinetic_energy

    # Plot the quantities
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    axes[0].plot(timestep * dt, kinetic_temperature, color='red')
    axes[0].axhline(y=kT, color='black', linestyle='--', label='Fixed T')
    axes[0].set_xlabel('Time [ps]')
    axes[0].set_ylabel('Kinetic Temperature [kJ/mol]')

    axes[1].plot(timestep * dt, pressure, color='blue')
    axes[1].axhline(y=np.mean(pressure), color='black', linestyle='--', label='Avg P')
    axes[1].set_xlabel('Time [ps]')
    axes[1].set_ylabel('Pressure')

    plt.tight_layout()
    plt.savefig(f'ThermodynamicQuantities_{params_name}.png')
    plt.close()

    # Save the thermodynamic quantities data to CSV
    thermodynamic_data = pd.DataFrame({
        'Time [ps]': timestep * dt,
        'Kinetic Temperature [kJ/mol]': kinetic_temperature,
        'Pressure': pressure
    })
    thermodynamic_data.to_csv(f'ThermodynamicQuantities_{params_name}.csv', index=False)

    # Plot the Energies
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    ax.plot(timestep * dt, potential_energy, label='Potential Energy', color='blue')
    ax.axhline(y=np.mean(potential_energy), color='blue', linestyle='--', label='Avg PE')
    ax.plot(timestep * dt, kinetic_energy, label='Kinetic Energy', color='green')
    ax.axhline(y=np.mean(kinetic_energy), color='green', linestyle='--', label='Avg KE')
    ax.plot(timestep * dt, total_energy, label='Total Energy', color='red')
    ax.axhline(y=np.mean(total_energy), color='red', linestyle='--', label='Avg Energy')
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Energy [kJ/mol]')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'Energies_{params_name}.png')
    plt.close()

    # Save the energy data to CSV
    energy_data = pd.DataFrame({
        'Time [ps]': timestep * dt,
        'Potential Energy [kJ/mol]': potential_energy,
        'Kinetic Energy [kJ/mol]': kinetic_energy,
        'Total Energy [kJ/mol]': total_energy
    })
    energy_data.to_csv(f'Energies_{params_name}.csv', index=False)

# Monitoring Ramp Traj
def monitorRamp(snapshot_file, kT, params_name, dt):
    # Read the logged data from the GSD file
    data = gsd.hoomd.read_log(snapshot_file)

    # Extract the quantities of interest from the data dictionary
    timestep = data['configuration/step']
    potential_energy = data['log/md/compute/ThermodynamicQuantities/potential_energy']
    kinetic_temperature = data['log/md/compute/ThermodynamicQuantities/kinetic_temperature']
    pressure = data['log/md/compute/ThermodynamicQuantities/pressure']
    kinetic_energy = data['log/md/compute/ThermodynamicQuantities/kinetic_energy']
    total_energy = potential_energy + kinetic_energy

    # Plot the quantities
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    axes[0].plot(timestep * dt, kinetic_temperature, color='red')
    axes[0].axhline(y=kT, color='black', linestyle='--', label='Fixed T')
    axes[0].set_xlabel('Time [ps]')
    axes[0].set_ylabel('Kinetic Temperature [kJ/mol]')

    axes[1].plot(timestep * dt, pressure, color='blue')
    axes[1].axhline(y=np.mean(pressure), color='black', linestyle='--', label='Avg P')
    axes[1].set_xlabel('Time [ps]')
    axes[1].set_ylabel('Pressure')

    plt.tight_layout()
    plt.savefig(f'ThermodynamicQuantities_Ramp_{params_name}.png')
    plt.close()

    # Save the thermodynamic quantities data to CSV
    thermodynamic_data = pd.DataFrame({
        'Time [ps]': timestep * dt,
        'Kinetic Temperature [kJ/mol]': kinetic_temperature,
        'Pressure': pressure
    })
    thermodynamic_data.to_csv(f'ThermodynamicQuantities_Ramp_{params_name}.csv', index=False)

    # Plot the Energies
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    ax.plot(timestep * dt, potential_energy, label='Potential Energy', color='blue')
    ax.axhline(y=np.mean(potential_energy), color='blue', linestyle='--', label='Avg PE')
    ax.plot(timestep * dt, kinetic_energy, label='Kinetic Energy', color='green')
    ax.axhline(y=np.mean(kinetic_energy), color='green', linestyle='--', label='Avg KE')
    ax.plot(timestep * dt, total_energy, label='Total Energy', color='red')
    ax.axhline(y=np.mean(total_energy), color='red', linestyle='--', label='Avg Energy')
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Energy [kJ/mol]')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'Energies_Ramp_{params_name}.png')
    plt.close()

    # Save the energy data to CSV
    energy_data = pd.DataFrame({
        'Time [ps]': timestep * dt,
        'Potential Energy [kJ/mol]': potential_energy,
        'Kinetic Energy [kJ/mol]': kinetic_energy,
        'Total Energy [kJ/mol]': total_energy
    })
    energy_data.to_csv(f'Energies_Ramp_{params_name}.csv', index=False)

# Monitoring 2 Traj
def monitor2(snapshot_file1, snapshot_file2, kT, params_name, dt):
    # Read the logged data from the GSD files
    data1 = gsd.hoomd.read_log(snapshot_file1)
    data2 = gsd.hoomd.read_log(snapshot_file2)

    # Extract the quantities of interest from the data dictionaries of both files
    timestep1 = data1['configuration/step']
    potential_energy1 = data1['log/md/compute/ThermodynamicQuantities/potential_energy']
    kinetic_temperature1 = data1['log/md/compute/ThermodynamicQuantities/kinetic_temperature']
    pressure1 = data1['log/md/compute/ThermodynamicQuantities/pressure']
    kinetic_energy1 = data1['log/md/compute/ThermodynamicQuantities/kinetic_energy']
    total_energy1 = potential_energy1 + kinetic_energy1

    timestep2 = data2['configuration/step']
    potential_energy2 = data2['log/md/compute/ThermodynamicQuantities/potential_energy']
    kinetic_temperature2 = data2['log/md/compute/ThermodynamicQuantities/kinetic_temperature']
    pressure2 = data2['log/md/compute/ThermodynamicQuantities/pressure']
    kinetic_energy2 = data2['log/md/compute/ThermodynamicQuantities/kinetic_energy']
    total_energy2 = potential_energy2 + kinetic_energy2

    # Combine data from both files
    timestep = np.concatenate((timestep1, timestep2))
    kinetic_temperature = np.concatenate((kinetic_temperature1, kinetic_temperature2))
    pressure = np.concatenate((pressure1, pressure2))
    potential_energy = np.concatenate((potential_energy1, potential_energy2))
    kinetic_energy = np.concatenate((kinetic_energy1, kinetic_energy2))
    total_energy = np.concatenate((total_energy1, total_energy2))

    # Plot the quantities
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    axes[0].plot(timestep * dt, kinetic_temperature, color='red')
    axes[0].axhline(y=kT, color='black', linestyle='--', label='Fixed T')
    axes[0].set_xlabel('Time [ps]')
    axes[0].set_ylabel('Kinetic Temperature [kJ/mol]')

    axes[1].plot(timestep * dt, pressure, color='blue')
    axes[1].axhline(y=np.mean(pressure), color='black', linestyle='--', label='Avg P')
    axes[1].set_xlabel('Time [ps]')
    axes[1].set_ylabel('Pressure')

    plt.tight_layout()
    plt.savefig(f'ThermodynamicQuantities_{params_name}.png')
    plt.close()

    # Save the thermodynamic quantities data to CSV
    thermodynamic_data = pd.DataFrame({
        'Time [ps]': timestep * dt,
        'Kinetic Temperature [kJ/mol]': kinetic_temperature,
        'Pressure': pressure
    })
    thermodynamic_data.to_csv(f'ThermodynamicQuantities_{params_name}.csv', index=False)

    # Plot the Energies
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    ax.plot(timestep * dt, potential_energy, label='Potential Energy', color='blue')
    ax.axhline(y=np.mean(potential_energy), color='blue', linestyle='--', label='Avg PE')
    ax.plot(timestep * dt, kinetic_energy, label='Kinetic Energy', color='green')
    ax.axhline(y=np.mean(kinetic_energy), color='green', linestyle='--', label='Avg KE')
    ax.plot(timestep * dt, total_energy, label='Total Energy', color='red')
    ax.axhline(y=np.mean(total_energy), color='red', linestyle='--', label='Avg Energy')
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Energy [kJ/mol]')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'Energies_{params_name}.png')
    plt.close()

    # Save the energy data to CSV
    energy_data = pd.DataFrame({
        'Time [ps]': timestep * dt,
        'Potential Energy [kJ/mol]': potential_energy,
        'Kinetic Energy [kJ/mol]': kinetic_energy,
        'Total Energy [kJ/mol]': total_energy
    })
    energy_data.to_csv(f'Energies_{params_name}.csv', index=False)

# Monitoring 3 Traj
def monitor3(snapshot_file1, snapshot_file2, snapshot_file3, kT, params_name, dt):
    # Read the logged data from the GSD files
    data1 = gsd.hoomd.read_log(snapshot_file1)
    data2 = gsd.hoomd.read_log(snapshot_file2)
    data3 = gsd.hoomd.read_log(snapshot_file3)

    # Extract the quantities of interest from the data dictionaries of both files
    timestep1 = data1['configuration/step']
    potential_energy1 = data1['log/md/compute/ThermodynamicQuantities/potential_energy']
    kinetic_temperature1 = data1['log/md/compute/ThermodynamicQuantities/kinetic_temperature']
    pressure1 = data1['log/md/compute/ThermodynamicQuantities/pressure']
    kinetic_energy1 = data1['log/md/compute/ThermodynamicQuantities/kinetic_energy']
    total_energy1 = potential_energy1 + kinetic_energy1

    timestep2 = data2['configuration/step']
    potential_energy2 = data2['log/md/compute/ThermodynamicQuantities/potential_energy']
    kinetic_temperature2 = data2['log/md/compute/ThermodynamicQuantities/kinetic_temperature']
    pressure2 = data2['log/md/compute/ThermodynamicQuantities/pressure']
    kinetic_energy2 = data2['log/md/compute/ThermodynamicQuantities/kinetic_energy']
    total_energy2 = potential_energy2 + kinetic_energy2

    timestep3 = data3['configuration/step']
    potential_energy3 = data3['log/md/compute/ThermodynamicQuantities/potential_energy']
    kinetic_temperature3 = data3['log/md/compute/ThermodynamicQuantities/kinetic_temperature']
    pressure3 = data3['log/md/compute/ThermodynamicQuantities/pressure']
    kinetic_energy3 = data3['log/md/compute/ThermodynamicQuantities/kinetic_energy']
    total_energy3 = potential_energy3 + kinetic_energy3

    # Combine data from both files
    timestep = np.concatenate((timestep1, timestep2, timestep3))
    kinetic_temperature = np.concatenate((kinetic_temperature1, kinetic_temperature2, kinetic_temperature3))
    pressure = np.concatenate((pressure1, pressure2, pressure3))
    potential_energy = np.concatenate((potential_energy1, potential_energy2, potential_energy3))
    kinetic_energy = np.concatenate((kinetic_energy1, kinetic_energy2, kinetic_energy3))
    total_energy = np.concatenate((total_energy1, total_energy2, total_energy3))

    # Plot the quantities
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    axes[0].plot(timestep * dt, kinetic_temperature, color='red')
    axes[0].axhline(y=kT, color='black', linestyle='--', label='Fixed T')
    axes[0].set_xlabel('Time [ps]')
    axes[0].set_ylabel('Kinetic Temperature [kJ/mol]')

    axes[1].plot(timestep * dt, pressure, color='blue')
    axes[1].axhline(y=np.mean(pressure), color='black', linestyle='--', label='Avg P')
    axes[1].set_xlabel('Time [ps]')
    axes[1].set_ylabel('Pressure')

    plt.tight_layout()
    plt.savefig(f'ThermodynamicQuantities_{params_name}.png')
    plt.close()

    # Save the thermodynamic quantities data to CSV
    thermodynamic_data = pd.DataFrame({
        'Time [ps]': timestep * dt,
        'Kinetic Temperature [kJ/mol]': kinetic_temperature,
        'Pressure': pressure
    })
    thermodynamic_data.to_csv(f'ThermodynamicQuantities_{params_name}.csv', index=False)

    # Plot the Energies
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    ax.plot(timestep * dt, potential_energy, label='Potential Energy', color='blue')
    ax.axhline(y=np.mean(potential_energy), color='blue', linestyle='--', label='Avg PE')
    ax.plot(timestep * dt, kinetic_energy, label='Kinetic Energy', color='green')
    ax.axhline(y=np.mean(kinetic_energy), color='green', linestyle='--', label='Avg KE')
    ax.plot(timestep * dt, total_energy, label='Total Energy', color='red')
    ax.axhline(y=np.mean(total_energy), color='red', linestyle='--', label='Avg Energy')
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Energy [kJ/mol]')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'Energies_{params_name}.png')
    plt.close()

    # Save the energy data to CSV
    energy_data = pd.DataFrame({
        'Time [ps]': timestep * dt,
        'Potential Energy [kJ/mol]': potential_energy,
        'Kinetic Energy [kJ/mol]': kinetic_energy,
        'Total Energy [kJ/mol]': total_energy
    })
    energy_data.to_csv(f'Energies_{params_name}.csv', index=False)

#These lines call the functions
#create_system(SpeciesList, box_length, seed, Time)
#run_energy_minimization('initial_PreRamp.gsd',EMin_dt,F_tol,AngMom_tol,E_tol,Min_steps,LJCut)
#Box_Ramp('initial_PreRamp.gsd', Box_Ramp_Factor, box_length,box_length_final, ramp_rate,ramp_dt, ramp_sim_steps, kT, Viscosity, EnergyScaling, dt, TotalTime, seed, time, LJCut, ramp_steps2fin, fast_trigger_rate,kappa)
#monitorRamp('initial_PostRamp.gsd', kT, params_name, dt)
#MSDanalysisRamp('initial_PostRamp.gsd', dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor)

#run_brownian_dynamics_simulation('initial_PostRamp.gsd',kT, Viscosity, EnergyScaling, dt, TotalTime, seed, time, LJCut, fast_trigger_rate,  slow_trigger_rate, switch_step, kappa)
extended_brownian_dynamics_simulation('initial_PostRamp.gsd', kT, Viscosity, EnergyScaling, dt, ext_trigger_rate, seed, time, LJCut,Extended_time, kappa)
#MSDanalysis2('trajectory_slow.gsd', 'trajectory_fast.gsd', dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor)
#monitor3('trajectory_slow.gsd', 'trajectory_fast.gsd', 'trajectory_extended.gsd', kT, params_name, dt)
#MSDanalysis3('trajectory.gsd', dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor)

#monitorRamp('initial_PostRamp.gsd', kT, params_name, dt)
#MSDanalysisRamp('initial_PostRamp.gsd', dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor)

#monitor1('trajectory.gsd', kT, params_name, dt)
#MSDanalysis1('trajectory.gsd', dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor)

#for species in Species2Monitor:
    #print(species)
    #RDFAnalysis('trajectory.gsd', [species], box_length)

#OldRDFAnalysis('trajectory.gsd', params_name, SpeciesList)

warnings.resetwarnings()