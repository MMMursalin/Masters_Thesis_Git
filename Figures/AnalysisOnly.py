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
    plt.savefig(f'ThermodynamicQuantities_Ramp_{params_name}.svg')
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
    plt.savefig(f'Energies_Ramp_{params_name}.svg')
    plt.close()

    # Save the energy data to CSV
    energy_data = pd.DataFrame({
        'Time [ps]': timestep * dt,
        'Potential Energy [kJ/mol]': potential_energy,
        'Kinetic Energy [kJ/mol]': kinetic_energy,
        'Total Energy [kJ/mol]': total_energy
    })
    energy_data.to_csv(f'Energies_Ramp_{params_name}.csv', index=False)

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
            plt.savefig(f'MSD_Ramp_{Species2Monitor}_{params_name}.svg')
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
            plt.savefig(f'Table_Ramp_{params_name}.svg', bbox_inches='tight', pad_inches=0.5)
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
    plt.savefig(f'ThermodynamicQuantities_{params_name}.svg')
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
    plt.savefig(f'Energies_{params_name}.svg')
    plt.close()

    # Save the energy data to CSV
    energy_data = pd.DataFrame({
        'Time [ps]': timestep * dt,
        'Potential Energy [kJ/mol]': potential_energy,
        'Kinetic Energy [kJ/mol]': kinetic_energy,
        'Total Energy [kJ/mol]': total_energy
    })
    energy_data.to_csv(f'Energies_{params_name}.csv', index=False)

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
            plt.savefig(f'MSD_{Species2Monitor}_{params_name}.svg')
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
                'Log Time': log_time_points.tolist(),
                'Log MSD ': log_msd_values.tolist()
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
            plt.savefig(f'Table_{params_name}.svg', bbox_inches='tight', pad_inches=0.5)
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
            'Log Time': species_data['Log Time'],
            'Log MSD ': species_data['Log MSD ']
        })
        df.to_csv(f'MSD_{species_name}_{params_name}.csv', index=False)

    print('CSV files saved for each species.')
    print('End of all MSD Analysis')
    print('--------------------')

#pairwise RDF Calculation
def CalcAllRDF(start_frame, end_frame, trajectory_file, file_path, Species2Monitor):
    # Initialize DataFrames to store RDF data for all frames
    rdf_data_all_frames = None
    norm_rdf_data_all_frames = None

    # Loop through each frame in the range [start_frame, end_frame]
    with gsd.hoomd.open(trajectory_file, 'r') as traj:
        for frame_number in range(start_frame, min(end_frame, len(traj))):
            frame = traj[frame_number]

            # Extract box length
            box_length = np.linalg.norm(frame.configuration.box[:3])

            # Extract particle positions and types
            positions = frame.particles.position
            type_ids = frame.particles.typeid
            type_dict = {i: t for i, t in enumerate(frame.particles.types)}

            # Load species compound data
            species_data = pd.read_csv(file_path)
            copies_dict = species_data.set_index('Name')['Copies'].to_dict()

            # Prepare DataFrames to store RDF data for the current frame
            rdf_data = pd.DataFrame()
            norm_rdf_data = pd.DataFrame()

            for species_i in Species2Monitor:
                for species_j in Species2Monitor:
                    # Select particles of the two species
                    mask_i = np.array([type_dict[tid] == species_i for tid in type_ids])
                    mask_j = np.array([type_dict[tid] == species_j for tid in type_ids])
                    positions_i = positions[mask_i]
                    positions_j = positions[mask_j] if species_i != species_j else positions_i

                    # Compute the RDF
                    rdf = freud.density.RDF(bins=500, r_max=box_length, normalize=True)
                    rdf.compute(system=(frame.configuration.box, positions_i), query_points=positions_j, reset=False)

                    # Create a DataFrame for this pair's RDF data
                    rdf_pair_data = pd.DataFrame({'r': rdf.bin_centers, f'RDF_{species_i}_{species_j}': rdf.rdf})
                    norm_rdf_pair_data = pd.DataFrame({f'RDF_{species_i}_{species_j}': rdf.rdf / copies_dict[species_j]})

                    # Aggregate RDF data across all frames
                    if rdf_data_all_frames is None:
                        rdf_data_all_frames = rdf_pair_data
                        norm_rdf_data_all_frames = norm_rdf_pair_data
                    else:
                        rdf_data_all_frames = rdf_data_all_frames.add(rdf_pair_data.iloc[:, 1:], fill_value=0)
                        norm_rdf_data_all_frames = norm_rdf_data_all_frames.add(norm_rdf_pair_data.iloc[:, 1:], fill_value=0)

    # Normalize the RDF data across all frames
    rdf_data_all_frames.iloc[:, 1:] = rdf_data_all_frames.iloc[:, 1:] / (end_frame - start_frame + 1)
    norm_rdf_data_all_frames.iloc[:, 1:] = norm_rdf_data_all_frames.iloc[:, 1:] / (end_frame - start_frame + 1)

    # Save the non-normalized data to a CSV file
    rdf_data_all_frames.to_csv('RDFs.csv', index=False)

    # Save the normalized data to a CSV file
    #norm_rdf_data_all_frames.to_csv('Norm_RDFs_all_frames.csv', index=False)

#Cluster Analysis
def CountClusters(start_frame, end_frame, trajectory_file, file_path):
    # Initialize lists to hold data for all frames
    composition_data_list = []
    stats_list = []

    with gsd.hoomd.open(trajectory_file, 'r') as traj:
        for frame_number in range(start_frame, min(end_frame + 1, len(traj))):
            frame = traj[frame_number]

            # Extract the box and positions
            box = frame.configuration.box
            positions = frame.particles.position

            # Convert HOOMD box to Freud box
            freud_box = freud.box.Box.from_box(box)

            # Initialize the freud cluster object
            cl = freud.cluster.Cluster()

            # Compute clusters
            cl.compute(system=(freud_box, positions), neighbors={'r_max': 1.0})

            # Filter out clusters of size 1 or smaller
            cluster_sizes = np.bincount(cl.cluster_idx)
            valid_clusters = np.nonzero(cluster_sizes > 1)[0]

            # Load species compound data
            species_data = pd.read_csv(file_path)
            mass_dict = species_data.set_index('Name')['Mass'].to_dict()
            charge_dict = species_data.set_index('Name')['Charge'].to_dict()
            radius_dict = species_data.set_index('Name')[' Radius'].to_dict()

            # Get the names of the particle types
            type_names = frame.particles.types

            # Prepare a list to collect data for each cluster
            frame_stats_list = []

            # Loop over each valid cluster
            for i in valid_clusters:
                # Find particle indices in the current cluster
                particle_indices = np.where(cl.cluster_idx == i)[0]
                # Get the types of particles in the current cluster
                particle_types = frame.particles.typeid[particle_indices]
                particle_type_names = [type_names[ptype] for ptype in particle_types]

                # Calculate the composition and statistics
                composition_counts = np.bincount(particle_types, minlength=len(type_names))

                # Calculate the statistics
                num_constituents = len(particle_indices)
                total_mass = sum(mass_dict[type_names[ptype]] for ptype in particle_types)
                total_charge = sum(charge_dict[type_names[ptype]] for ptype in particle_types)
                sum_radii = sum(radius_dict[type_names[ptype]] for ptype in particle_types)

                # Append the statistics to the list for the frame
                frame_stats_list.append({
                    'Cluster': f'Frame{frame_number}_Cluster{i}',
                    'Num_Constituents': num_constituents,
                    'Total_Mass': total_mass,
                    'Total_Charge': total_charge,
                    'Sum_Radii': sum_radii
                })

            # Add the frame's composition and statistics data to the main lists
            stats_list.extend(frame_stats_list)

    # Convert the list of dictionaries to a DataFrame for statistics
    stats_data = pd.DataFrame(stats_list)

    # There is no direct composition comparison across frames, as compositions are inherently per-frame.
    # If needed, further processing can be done outside this function to analyze composition trends over frames.

    # Save the statistics to a CSV file
    stats_data.to_csv('Cluster_Statistics.csv')

    # Returning stats_data; composition_data concept is reframed as it's not directly comparable across frames
    return stats_data

#monitorRamp('initial_PostRamp.gsd', kT, params_name, dt)
#MSDanalysisRamp('initial_PostRamp.gsd', dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor)

monitor1('trajectory.gsd', kT, params_name, dt)
MSDanalysis1('trajectory.gsd', dt, kT, Viscosity, Traj_Write_Rate, params_name, SpeciesList, Species2Monitor)


CalcAllRDF( 1, 50, 'trajectory.gsd', file_path, Species2Monitor)
CountClusters(1, 50, 'trajectory.gsd', file_path)





warnings.resetwarnings()
