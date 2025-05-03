
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import ray
import matplotlib.pyplot as plt
import numpy as np
from model import LUT_Model as LUT  # From provided model files

from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
from helper import firing_rate

# Initialize Ray with multiple workers
ray.init(ignore_reinit_error=True, num_cpus=4)

# Constants
TARGET_VOLUME_RATIO = 0.95
MAX_PYR_FIRING_RATE = 50  # Hz (from neuro literature)
LUT_SIMULATION_TIME = 8 * 60 * 60  # seconds

# Track results for plotting
trial_numbers = []
pyr_firing_rates = []
parameter_values = []
a1 = []
a2 = []
f1 = []
f2 = []
injected_power = []
volume_deviations = []
max_volumes = []
normalized_controls = []


# Define the evaluation function
def evaluate_parameters(parameters):
    # Extract parameters
    amp1 = parameters["amp1"]
    amp2 = parameters["amp2"]
    freq1 = parameters["freq1"]
    freq2 = parameters["freq2"]

    total_time = 500  # ms

    # Run simulations in parallel
    results = [
        simulation_Pyr.remote(
            num_electrode=1,
            amp1=amp1, amp2=amp2, freq1=freq1, freq2=freq2,
            total_time=total_time,
            plot_waveform=False
        ),
        simulation_PV.remote(
            num_electrode=1,
            amp1=amp1, amp2=amp2, freq1=freq1, freq2=freq2,
            total_time=total_time,
            plot_waveform=False
        )
    ]

    # Get results
    (response_Pyr, t), (response_PV, t) = ray.get(results)

    # Calculate firing rates
    FR_Pyr = firing_rate(response_Pyr, total_time)

    # Normalize pyramidal firing rate for bladder control
    normalized_control = np.clip(FR_Pyr / MAX_PYR_FIRING_RATE, 0.0, 1.0)

    # Run LUT simulation with normalized control
    lut_model = LUT.LUT()
    lut_model.set_manual_control(normalized_control)

    df = lut_model.process_neural_input(
        seed=42,
        maxTime=LUT_SIMULATION_TIME,
        dT=0.1,
        trigger_metric='volume',
        verbose=False
    )

    # Calculate volume deviation metrics
    target_volume = TARGET_VOLUME_RATIO * lut_model.bladder_args['max_V_B']
    post_trigger = df[df['V_B'] >= target_volume]

    if not post_trigger.empty:
        avg_deviation = np.mean(np.abs(post_trigger['V_B'] - target_volume))
        max_deviation = np.max(post_trigger['V_B']) - target_volume
    else:
        avg_deviation = lut_model.bladder_args['max_V_B']  # Max possible deviation
        max_deviation = lut_model.bladder_args['max_V_B']
    power = 0.5*(amp1 ** 2 + amp2 ** 2)
    return {
        "volume_deviation": (avg_deviation + 0.3 * max_deviation),
        "FR_Pyr": FR_Pyr,
        "normalized_control": normalized_control,
        "max_volume": df['V_B'].max(),
        "injected_power": power
    }

generation_strategy = GenerationStrategy(
    name="ThompsonSamplingStrategy",
    steps=[
        GenerationStep(  # Initial exploration
            model=Models.SOBOL,
            num_trials=10,
            max_parallelism=5
        ),
        GenerationStep(  # Thompson sampling phase
            model=Models.THOMPSON,
            num_trials=-1,
            model_kwargs={"n_sobol_samples": 500}
        )
    ]
)
# Initialize Ax client
ax_client = AxClient(generation_strategy=generation_strategy)

# Define the optimization problem
ax_client.create_experiment(
    name="integrated_bladder_control",
    parameters=[
        {
            "name": "amp1",
            "type": "range",
            "bounds": [50.0, 200.0],
            "value_type": "float",
        },
        {
            "name": "amp2",
            "type": "range",
            "bounds": [50.0, 200.0],
            "value_type": "float",
        },
        {
            "name": "freq1",
            "type": "range",
            "bounds": [20.0, 100.0],
            "value_type": "float",
        },
        {
            "name": "freq2",
            "type": "range",
            "bounds": [20.0, 100.0],
            "value_type": "float",
        },
    ],
    objectives={
        "injected_power": ObjectiveProperties(minimize=True),
    },
)
ax_client.add_tracking_metrics(["FR_Pyr", "volume_deviation", "max_volume",  "normalized_control"])
# Run optimization
num_trials = 20
for i in range(num_trials):
    parameters, trial_index = ax_client.get_next_trial()
    results = evaluate_parameters(parameters)
    # print(f"************* Trial {results["max_volume"]*10**6}")
    if results["FR_Pyr"] == 0 or (round(results["max_volume"] * 10 ** 6, 0)) <= 480:
        # Log this trial as a failure so Ax ignores it in optimization
        ax_client.log_trial_failure(trial_index=trial_index)
        continue
    # Store results for plotting
    trial_numbers.append(i + 1)
    pyr_firing_rates.append(results["FR_Pyr"])
    parameter_values.append(parameters)
    a1.append(parameters["amp1"])
    a2.append(parameters["amp2"])
    f1.append(parameters["freq1"])
    f2.append(parameters["freq2"])
    injected_power.append(0.5 * (pow(parameters["amp1"], 2) + pow(parameters["amp2"], 2)))
    volume_deviations.append(results["volume_deviation"])
    max_volumes.append(results["max_volume"])
    normalized_controls.append(results["normalized_control"])

    # Complete the trial
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data=results
    )

# Get best parameters
best_parameters, metrics = ax_client.get_best_parameters()
print("\nBest parameters found:")
print(f"A1 (amp1): {best_parameters.get('amp1'):.4f}")
print(f"A2 (amp2): {best_parameters.get('amp2'):.4f}")
print(f"f1 (freq1): {best_parameters.get('freq1'):.4f}")
print(f"f2 (freq2): {best_parameters.get('freq2'):.4f}")
print(f"Minimum volume deviation: {metrics[0].get('volume_deviation'):.4e} m³")
print(f"Maximum volume: {metrics[0].get('max_volume'):.4e} m³")
print(f"Normalized control: {metrics[0].get('normalized_control'):.4e} Hz")
print(f"Power: {metrics[0].get('injected_power'):.4e} mW")



# Plotting functions
def plot_neural_results():
    plt.figure(figsize=(15, 10))

    plt.plot(trial_numbers, pyr_firing_rates, 'r-o', label='Pyramidal Neuron (gPyr)')
    plt.xlabel('Trial Number')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('Neuron Firing Rates Across Optimization Trials')
    plt.legend()
    plt.grid(True)


def plot_stimulation_parameters():
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(trial_numbers, a1, 'b-o')
    plt.xlabel('Trial Number')
    plt.ylabel('Amplitude 1 (mV)')
    plt.title('A1 Across Optimization Trials')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(trial_numbers, a2, 'g-o')
    plt.xlabel('Trial Number')
    plt.ylabel('Amplitude 2 (mV)')
    plt.title('A2 Across Optimization Trials')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(trial_numbers, f1, 'r-o')
    plt.xlabel('Trial Number')
    plt.ylabel('F1 (Hz)')
    plt.title('F1 Across Optimization Trials')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(trial_numbers, f2, 'm-o')
    plt.xlabel('Trial Number')
    plt.ylabel('Frequency 2 (Hz)')
    plt.title('F2 Across Optimization Trials')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_bladder_metrics():
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.plot(trial_numbers, [v * 1e6 for v in max_volumes], 'm-o')  # Convert to ml
    plt.xlabel('Trial Number')
    plt.ylabel('Max Volume (ml)')
    plt.title('Maximum Bladder Volume Reached')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(trial_numbers, normalized_controls, 'c-o')
    plt.xlabel('Trial Number')
    plt.ylabel('Normalized Control')
    plt.title('Pyramidal->Bladder Control Signal')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(trial_numbers, volume_deviations, 'k-o')
    plt.xlabel('Trial Number')
    plt.ylabel('Volume Deviation (m³)')
    plt.title('Bladder Control Performance')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(trial_numbers, injected_power, 'y-o')
    plt.xlabel('Trial Number')
    plt.ylabel('Injected Power (mW)')
    plt.title('Stimulation Power Consumption')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate all plots
    plot_neural_results()
    plot_stimulation_parameters()
    plot_bladder_metrics()

    # Shutdown Ray
    ray.shutdown()
