"""
Some code from https://github.com/peter-i-frazier/group-testing
"""
import numpy as np
import pickle
import sys
from joblib import Parallel, delayed

from multi_group_simulation import MultiGroupSimulation
from load_params import load_params
from scipy.stats import qmc

sys.path.append(sys.path[0][:-len('data/covid/')])  # for imports to work
print(sys.path)

from core.utils import cross_product, construct_grid_1d

base_dir = 'data/covid/'


def get_cum_inf_trajectory(df):
    return np.sum(df[['cumulative_mild', 'cumulative_severe']], axis=1)


def get_trajectories(transmission_p, n, params_list, interaction_matrix, group_names, T=19 * 7):
    for idx in range(3):
        params_list[idx]['exposed_infection_p'] = transmission_p

    sim = MultiGroupSimulation(params_list, interaction_matrix, group_names)

    trajectories = list()

    for _ in range(n):
        sim.run_new_trajectory(T)
        group_results = []
        for group in sim.sims:
            df = group.sim_df
            group_results.append(df)
        trajectories.append(group_results)

    return trajectories


def median_cases(prop_tests_0, prop_tests_1, prop_init_cases_0, prop_init_cases_1,
                 base_transmission_offset,
                 interaction_offset,
                 idx_track,
                 num_trajectories=10):
    """

    :param prop_tests_0: float in [0, 1]. prop_tests_0 + prop_tests_1 <= 1.
    :param prop_tests_1: float in [0, 1]. prop_tests_0 + prop_tests_1 <= 1.
    :param prop_init_cases_0: float in [0, 1]. Environment variable.
    :param prop_init_cases_1: float in [0, 1]. Environment variable.
    :param base_transmission_offset: float in [0, 1]. Environment variable.
    :param interaction_offset: float in [0, 1]. Environment variable.
    :param idx_track. For tracking
    :param num_trajectories: int. Number of Monte Carlo trajectories to run.
    :return: float. Median number of cases.
    """
    # ==== Set up parameters ====

    # Loading group params
    params = load_params(base_dir + 'group_1_students_post_movein_private.yaml')[1]
    params_list = [params.copy(), params.copy(), params.copy()]

    interaction_matrix = (10 + interaction_offset * 5) * np.array([[1, 0, 0],
                                                                   [0, 1, 0],
                                                                   [0, 0, 1]])

    # adding population size
    params_list[0]['population_size'] = 10000
    params_list[1]['population_size'] = 10000
    params_list[2]['population_size'] = 10000

    # total tests
    total_tests = 5000

    for idx in range(3):
        params_list[idx]['daily_outside_infection_p'] *= 0
        params_list[idx]['expected_contacts_per_day'] = interaction_matrix[idx][idx]

    # Set number of initial cases
    params_list[0]['initial_ID_prevalence'] = prop_init_cases_0 / 8
    params_list[1]['initial_ID_prevalence'] = prop_init_cases_1 / 8
    params_list[2]['initial_ID_prevalence'] = (1 - prop_init_cases_0 - prop_init_cases_1) / 8

    num_tests_0 = prop_tests_0 * total_tests
    num_tests_1 = prop_tests_1 * total_tests
    num_tests_2 = (1 - prop_tests_0 - prop_tests_1) * total_tests

    # Set number of tests for each population
    params_list[0]['test_population_fraction'] = num_tests_0 / params_list[0]['population_size']
    params_list[1]['test_population_fraction'] = num_tests_1 / params_list[1]['population_size']
    params_list[2]['test_population_fraction'] = num_tests_2 / params_list[2]['population_size']

    group_names = ['Pop. 1', 'Pop. 2', 'Pop. 3']

    # ==== Run simulations ====

    base_transmission_p = 0.1 + 0.1 * base_transmission_offset

    trajectories = get_trajectories(base_transmission_p, num_trajectories, params_list, interaction_matrix, group_names,
                                    T=19 * 7)

    cases = []
    for sim_run in trajectories:
        cases.append(list(
            get_cum_inf_trajectory(sim_run[0]) + get_cum_inf_trajectory(sim_run[1]) + get_cum_inf_trajectory(
                sim_run[2]))[-1])

    print(f"{idx_track} completed")
    return np.median(cases)


if __name__ == "__main__":
    action_density = 10
    context_density = 10
    action_dims = 2
    context_dims = 4
    action_lowers = [0] * action_dims
    action_uppers = [1] * action_dims
    context_lowers = [0] * context_dims
    context_uppers = [1] * context_dims

    dims = action_dims + context_dims
    lowers = action_lowers + context_lowers
    uppers = action_uppers + context_uppers
    num_samples = 2 ** 16

    sampler = qmc.Sobol(d=dims, scramble=False)
    sample = sampler.random(num_samples)
    scaled_samples = qmc.scale(sample, lowers, uppers)
    # Delete infeasible samples
    valid_samples = np.delete(scaled_samples, np.where(np.sum(scaled_samples[:, :2], axis=1) > 1), axis=0)
    valid_samples = np.delete(valid_samples, np.where(np.sum(valid_samples[:, 2:4], axis=1) > 1), axis=0)
    all_params = valid_samples
    print(f"Length of all_params: {all_params}")

    # # Action space
    # action_points = construct_grid_1d(action_lowers[0], action_uppers[0], action_density)
    # for i in range(action_dims - 1):
    #     action_points = cross_product(action_points,
    #                                   construct_grid_1d(action_lowers[i + 1], action_uppers[i + 1],
    #                                                     action_density))
    #
    # # Some actions are invalid: we only keep those that x[0] + x[1] <= 1
    # action_points = np.delete(action_points, np.where(np.sum(action_points, axis=1) > 1), axis=0)
    #
    # # Context space
    # context_points = construct_grid_1d(context_lowers[0], context_uppers[0], context_density)
    # for i in range(context_dims - 1):
    #     context_points = cross_product(context_points,
    #                                    construct_grid_1d(context_lowers[i + 1], context_uppers[i + 1],
    #                                                      context_density))
    # # Some contexts are invalid: we only keep those that c[0] + c[1] <= 1
    # context_points = np.delete(context_points, np.where(np.sum(context_points[:, :2], axis=1) > 1), axis=0)
    #
    # all_params = cross_product(action_points, context_points)

    all_results = Parallel(n_jobs=32)(delayed(median_cases)(*all_params[i], i) for i in range(len(all_params)))

    pickle.dump((all_params, all_results), open(base_dir + 'covid_params_results.p', 'wb'))

    X = all_params
    y = np.array(all_results)
    y_hat = ((-y - np.mean(-y)) / np.std(-y))[:, None]
    pickle.dump((X, y_hat), open(base_dir + 'covid_X_y.p', 'wb'))
