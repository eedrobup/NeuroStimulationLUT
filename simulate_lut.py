import argparse
import matplotlib.pyplot as plt
from model import LUT_Model as model

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run LUT model simulation with specified parameters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation.")
    parser.add_argument("--dT", type=float, default=0.1, help="Time step (s).")
    parser.add_argument("--omega_s", type=float, required=True, help="Manual control parameter omega_s.")

    args = parser.parse_args()

    seed = args.seed
    dT = args.dT
    omega_s = args.omega_s
    day = 10 * 60 * 60
    maxTime = 1 * day

    # Create a lower urinary tract instance
    LUT = model.LUT()
    LUT.set_manual_control(omega_s)

    # Run the simulation
    data = LUT.process_neural_input(maxTime, dT, seed=seed, p_unit='cmH2O', V_unit='ml')

    # Output results
    print(data)
    print(f'max Volume: {max(data["V_B"])}')
    print(f'Voiding: {max(data["voiding"][2:-1])}')

    # Plot bladder and sphincter pressure
    plt.plot(data['t'], data['p_D'], label="Bladder Pressure")
    plt.plot(data['t'], data['p_S'], label="Sphincter Pressure")
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (cmH2O)')
    plt.legend()
    plt.show()

    # Plot bladder volume
    plt.plot(data['t'], data['V_B'])
    plt.xlabel('Time (s)')
    plt.ylabel('Bladder Volume (mL)')
    plt.show()

    # Plot pressure gradient and voiding state
    plt.plot(data['t'], data['p_D'] - data['p_S'], label="Pressure Gradient")
    plt.plot(data['t'], data['voiding'], label="Voiding state")
    plt.xlabel('Time (s)')
    plt.ylabel('Voiding')
    plt.legend()
    plt.show()
