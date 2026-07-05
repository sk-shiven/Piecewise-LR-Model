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
    phases = ['phase1.py', 'phase2.py', 'phase3.py', 'phase4.py', 'phase5.py']
    for phase in phases:
        run_phase(phase)
    print("All phases completed successfully. The model has been tested on the validation dataset.")
