import subprocess
import sys

def run_phase(script_name):
    print(f"=====================================")
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {script_name}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"Success. Output:\n{result.stdout}")
        print(f"=====================================\n")

if __name__ == "__main__":
    phases = ['data_visualisation.py', 'discontinuous_splined_lr.py', 'splined_lr.py', 'Hyperparameter_Tuning_Loop.py', 'piecewise_linear_regression.py']
    for phase in phases:
        run_phase(phase)
    print("All phases completed successfully. The model has been tested on the validation dataset.")
