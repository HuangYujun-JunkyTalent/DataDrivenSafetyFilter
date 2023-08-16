from itertools import product

import numpy as np
import matplotlib.pyplot as plt


from simulators.simulation_settings import SafetyFilterTypes, SimulationInputRule
from simulators.single_curvature_simulator import SingleCurvatureSimulator


def main():
    simualte_type = 'SINGLE_CURVATURE'

    Simulator = {
        'SINGLE_CURVATURE'  : SingleCurvatureSimulator,
        # 'TRACK_SIMULATE'    : track_simulate,
    }.get(simualte_type)

    simulator = Simulator()
    
    random_seeds = [0]
    filter_types = [
        SafetyFilterTypes.INDIRECT_FITTING_TERMINAL,
        SafetyFilterTypes.INDIRECT_FIX_MU,
        SafetyFilterTypes.INDIRECT_ZERO_V,
        SafetyFilterTypes.INDIRECT_STOP,
        ]
    filter_params = {}
    simulation_input_rules = [SimulationInputRule.MAX_THROTTLE_SINE_STEER, SimulationInputRule.SINE_WAVE]
    dict_results = simulator.simulate_multi(random_seeds, filter_types, filter_params, simulation_input_rules)
    
    # plot global trajectory and track
    n_rows, n_cols = len(filter_types), len(simulation_input_rules)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, figsize=(8*n_rows,8*n_cols))
    gen = simulator.get_track_generator(density=300)
    for i, j in product(range(n_rows), range(n_cols)):
        results = dict_results[(random_seeds[0], filter_types[i], simulation_input_rules[j])][0]
        results.plot_vehicle_trajectory(ax=axs[i,j], gen=gen)
    plt.show(block=False)

    # choose one result to plot detailed rajectories
    results = dict_results[(random_seeds[0], filter_types[0], simulation_input_rules[0])][0]

    # plot given error dynamics trajectory
    plt.figure(figsize=(15,16))
    #velocity input
    ax = plt.subplot(221)
    results.plot_error_trajectory(0, ax, constraint=(-simulator.track_width/2, simulator.track_width/2))
    results.plot_predicted_error_slices(0, ax)
    ax = plt.subplot(222)
    results.plot_error_trajectory(1, ax, constraint=(simulator.mu_min*180/np.pi, simulator.mu_max*180/np.pi))
    results.plot_predicted_error_slices(1, ax)
    ax = plt.subplot(223)
    results.plot_error_trajectory(2, ax)
    results.plot_predicted_error_slices(2, ax)
    plt.show(block=False)

    # plot given and applied inputs
    plt.figure(figsize=(14,6))
    ax = plt.subplot(121)
    results.plot_input_applied(0, ax, constraint=(simulator.a_min, simulator.a_max))
    results.plot_input_obj(0, ax)
    ax = plt.subplot(122)
    results.plot_input_applied(1, ax, constraint=(-simulator.delta_max, simulator.delta_max))
    results.plot_input_obj(1, ax)
    plt.show(block=True)

    # ensure all figs are closed before return
    while plt.get_fignums():
        pass
    print('All figures closed')

if __name__ == "__main__":
    main()
