from operator import indexOf

from model import LUT_Model as model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    seed = 42
    dT = 0.1
    day = 10 * 60 * 60
    maxTime = 1 * day

    # Create a lower urinary tract instance
    LUT = model.LUT()
    # LUT.set_full_bladder()
    LUT.set_manual_control(0.0625)
    # Run the simulation
    data = LUT.process_neural_input(maxTime, dT, seed=seed, p_unit='cmH2O', V_unit='ml')

    # Plot the results
    print(data)
    print(f'max Volume: {max(data['V_B'])}')
    print(f'Voiding: {max(data['voiding'][2:-1])}')


"""
    plt.plot(data['t'], data['p_D'],label="Bladder Pressure")
    plt.plot(data['t'], data['p_S'],label="Sphincter Pressure")
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (cmH2O)')
    plt.show()
    plt.plot(data['t'], data['V_B'])
    plt.xlabel('Time (s)')
    plt.ylabel('Bladder Volume (mL)')
    plt.show()

    plt.plot(data['t'],data['p_D'] - data['p_S'],label="Pressure Gradient")
    plt.plot(data['t'],(data['voiding']),label="Voiding state")
    plt.xlabel('Time (s)')
    plt.ylabel('Voiding')
    plt.legend()
    plt.show()"""