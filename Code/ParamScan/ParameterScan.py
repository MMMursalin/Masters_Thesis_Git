import os
import shutil
import subprocess
import itertools

# Create the main workspace folder
workspace_folder = "Workspace"
os.makedirs(workspace_folder, exist_ok=True)

# Define parameter combinations
#Vol_Frac_values = [0.5,0.6,0.7,0.8,0.9,1]
#EnergyScaling_values = [0,0.5,1,1.5,2]
Vol_Frac_values = [0.75]
EnergyScaling_values = [1]

parameter_combinations = list(itertools.product(Vol_Frac_values, EnergyScaling_values))

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Iterate over parameter combinations
for i, params in enumerate(parameter_combinations):
    # Create parameter-specific subfolder
    subfolder_name = f"Params-VolFrac_{params[0]}-Epsilon_{params[1]}"
    subfolder_path = os.path.join(workspace_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Copy input.py to the subfolder
    input_file = os.path.join(script_dir, "input.py")
    shutil.copyfile(input_file, os.path.join(subfolder_path, "input.py"))

    # Modify input.py with correct parameters
    input_file_path = os.path.join(subfolder_path, "input.py")
    with open(input_file_path, "r") as f:
        lines = f.readlines()
    with open(input_file_path, "w") as f:
        for line in lines:
            if line.startswith("Vol_Frac"):
                f.write(f"Vol_Frac = {params[0]}\n")
            elif line.startswith("EnergyScaling"):
                f.write(f"EnergyScaling = {params[1]}\n")
            else:
                f.write(line)

    # Copy remaining to the subfolders
    shutil.copyfile("Simulation.py", os.path.join(subfolder_path, "Simulation.py"))
    shutil.copyfile("submitter.sh", os.path.join(subfolder_path, "submitter.sh"))
    shutil.copyfile("jobfile.jdf", os.path.join(subfolder_path, "jobfile.jdf"))
    shutil.copyfile("run.sh", os.path.join(subfolder_path, "run.sh"))

    # Change directory to the subfolder
    os.chdir(subfolder_path)

    # Run Simulation.py and capture shell output
    output_file = f"output-VolFrac_{params[0]}-Epsilon_{params[1]}.txt"
    with open(output_file, "w") as f:
        subprocess.call(["bash", "submitter.sh"], stdout=f)

    # Change directory back to the main workspace folder
    os.chdir(script_dir)
