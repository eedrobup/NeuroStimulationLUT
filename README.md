## Neuron Model 

For running the code: 
1. Run the following command in the terminal: sh init_setup_run_once.sh
2. Run starter.py to perform a simulation and verify that everything is working correctly.
3. The first time you run a neuron simulation, the program may take longer to generate the initial save state. Subsequent simulations will run faster.


Common Issues: 
1. To run the simulations you need the NEURON software. Installing the NEURON software can be tricky. Please refer to their setup guide: https://www.neuron.yale.edu/neuron/download, in case init_setup_run_once.sh is not able to install neuron using pip


## LUT Model 
0. Install files in the requirements.txt folder if not already installed. 

## Run our test code 
Before execuring the code make sure that the directory contains Project_BO.py and Project_Thompson.py files in addition to the models folder containing the LUT model. 

1a. run Baysian Optimization using the command: *python3 Project_BO.py*
1b. run Thompson sampling using the command: *python3 Project_Thompson.py* 

For teseting any other omega_s values against LUT model use the command:
```python simulate_lut.py --omega_s <value> [--seed <int>] [--dT <float>]```
Parameters:
- --omega_s (required): Sphincter control parameter (e.g., 0.0625).
- --seed (optional): Random seed for reproducibility (default: 42).
- --dT (optional): Time step in seconds (default: 0.1).

Outputs:
- Prints time-series data for bladder pressure, sphincter pressure, volume, and voiding state.
- Displays three plots:
    - Bladder vs. Sphincter Pressure
    - Bladder Volume
    - Pressure Gradient & Voiding State