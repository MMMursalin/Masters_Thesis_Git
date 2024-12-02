import itertools
import os
import signac
import numpy as np
import time
import pandas as pd

### Define a list of species
# Matrix of spieces
#[Name,Charge,Mass,n_compounds,radius]
# Load the CSV file into a DataFrame
file_path = "CytoSys.csv"
SpeciesList_data = pd.read_csv(file_path, header=0, names=['Name', 'Charge', 'Mass', 'n_compounds', 'radius'])

# Create a list of lists from the DataFrame
SpeciesList = SpeciesList_data[['Name', 'Charge', 'Mass', 'n_compounds', 'radius']].values.tolist()

# Print the data_as_list to verify the format
#print(SpeciesList)

# List of species names to be analyzed
Species2Monitor = ['pyk','rp','acka','fusa','fba','parg','ace','etoh']

# Initialize an NxN interaction matrix with all elements set to 1
#row_count = len(SpeciesList)
#InteractionList = [[1 for _ in range(row_count)] for _ in range(row_count)] #All 1's for interactions

# Path to the interaction matrix CSV file
interaction_matrix_path = 'InteractionMatrix.csv'
interaction_matrix_df = pd.read_csv(interaction_matrix_path, header=None)
interaction_matrix_values = interaction_matrix_df.iloc[1:, 1:]
InteractionList = interaction_matrix_values.values.tolist()


###Create System Parameters
Vol_Frac = 0.042 #unitless fraction, 0.958 is water
# Debug: Print the contents of the SpeciesList DataFrame
print(SpeciesList)

#Occ_Vol = np.pi * (4/3) * np.sum([ radius**3 * n_compounds for _, _, _, n_compounds, radius in SpeciesList]) #nm**3
Occ_Vol = np.pi * (4/3) * np.sum(np.array(SpeciesList)[:, 4].astype(float)**3 * np.array(SpeciesList)[:, 3].astype(int))  # nm**3

###Box Ramping parameters
Box_Ramp_Factor = 3 # Factor that multiplies box length to then compress to final size, nm will affect volume as nm**3 i.e 2**(1/3)(len)=2(vol)
box_length = ((Occ_Vol/Vol_Frac)**(1/3)) * Box_Ramp_Factor #nm
box_length_final = box_length/Box_Ramp_Factor #Define the final box length so that it is from the volume fraction, nm
ramp_dt = 1E-4
ramp_steps2fin = int(1E4) #Timesteps to ramp from 0 to 1
ramp_sim_steps = int(ramp_steps2fin*1000) #Number of timesteps to run during ramp
ramp_rate = 1 # Rate at which box is rescaled

###Simulation Parameters
kT = 1 #KJ/mol
Viscosity = 1
I = 150000 #nM, Ionic Strength, 150mM in bio
kappa = 1/0.784 #nm-1, inverse Debye length, used when kappa isnt calculated
EnergyScaling = 1 #unitless

dt = 1E-4 #,picoseconds, 1E-4=0.1fs

TotalTime = 1E6 # .5ns ,number of steps 0.1fs*1E16=1s
Traj_Write_Rate= int(1E3) #Write output every Traj_Write_Rate step, TotalTime/Traj_Write_Rate~1E3 is fast

slow_trigger_rate = int(1E1) #Used to monitor the fastest Dynamics, unitless
switch_step = int(1E3) # Switch from slow to fast writer after this many steps, slow_trigger_rate/switch_step ~1E3, unitless

fast_trigger_rate = int(1E3)

Extended_time = int(5E9)  #
ext_trigger_rate = int(1E6)


Time = time.time() #The real time IRL
seed = int(np.round(Time)) #Round time to a int for seed
LJCut = False # Use cutoff LJ param (repulsive only)

params_name = f'trajectory_Epsilon={EnergyScaling}_kT={kT}_VolFrac={np.round(box_length, 1)}_{time.ctime()}'
