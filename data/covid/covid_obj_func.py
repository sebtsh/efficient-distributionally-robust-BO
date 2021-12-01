"""
Some code from https://github.com/peter-i-frazier/group-testing
"""
import numpy as np
import pickle
import sys
from joblib import Parallel, delayed

from multi_group_simulation import MultiGroupSimulation
from load_params import load_params

sys.path.append(sys.path[0][:-len('data/covid/')])  # for imports to work
print(sys.path)

from core.utils import cross_product


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
                 idx,
                 num_trajectories=10):
    """

    :param prop_tests_0: float in [0, 1]. prop_tests_0 + prop_tests_1 <= 1.
    :param prop_tests_1: float in [0, 1]. prop_tests_0 + prop_tests_1 <= 1.
    :param prop_init_cases_0: float in [0, 1]. Environment variable.
    :param prop_init_cases_1: float in [0, 1]. Environment variable.
    :param idx. For tracking
    :param num_trajectories: int. Number of Monte Carlo trajectories to run.
    :return: float. Median number of cases.
    """
    # ==== Set up parameters ====

    # Loading group params
    ug_ga_params = load_params(base_dir + 'group_1_students_post_movein_private.yaml')[1]
    ug_other_params = load_params(base_dir + 'group_2_students_post_movein_private.yaml')[1]
    gs_params = load_params(base_dir + 'group_3_students_post_movein_private.yaml')[1]
    params_list = [ug_ga_params.copy(), ug_other_params.copy(), gs_params.copy()]

    interaction_matrix = 10 * np.array([[3/4, 1/4, 1/4],
                                        [1/4, 3/4, 1/4],
                                        [1/4, 1/4, 3/4]])

    # adding population size
    params_list[0]['population_size'] = 10000
    params_list[1]['population_size'] = 10000
    params_list[2]['population_size'] = 10000

    # total tests
    total_tests = 5000

    for idx in range(3):
        params_list[idx]['daily_outside_infection_p'] *= 2
        params_list[idx]['expected_contacts_per_day'] = interaction_matrix[idx][idx]

    # Set number of initial cases
    params_list[0]['initial_ID_prevalence'] = prop_init_cases_0
    params_list[1]['initial_ID_prevalence'] = prop_init_cases_1
    params_list[2]['initial_ID_prevalence'] = 0.25

    num_tests_0 = prop_tests_0 * total_tests
    num_tests_1 = prop_tests_1 * total_tests
    num_tests_2 = (1 - prop_tests_0 - prop_tests_1) * total_tests

    # Set number of tests for each population
    params_list[0]['test_population_fraction'] = num_tests_0 / params_list[0]['population_size']
    params_list[1]['test_population_fraction'] = num_tests_1 / params_list[1]['population_size']
    params_list[2]['test_population_fraction'] = num_tests_2 / params_list[2]['population_size']

    group_names = ['Pop. 1', 'Pop. 2', 'Pop. 3']

    # ==== Run simulations ====

    base_transmission_p = 0.1

    trajectories = get_trajectories(base_transmission_p, num_trajectories, params_list, interaction_matrix, group_names,
                                    T=19 * 7)

    cases = []
    for sim_run in trajectories:
        cases.append(list(
            get_cum_inf_trajectory(sim_run[0]) + get_cum_inf_trajectory(sim_run[1]) + get_cum_inf_trajectory(
                sim_run[2]))[-1])

    print(f"{idx} completed")
    return np.median(cases)


if __name__ == "__main__":
    action_density = 10
    context_density = 10

    action_pairs = []
    for i in np.linspace(0, 1, action_density):
        for j in np.linspace(0, 1, action_density):
            if i + j <= 1:
                action_pairs.append([i, j])
    action_pairs = np.array(action_pairs)

    context_pairs = cross_product(np.linspace(0, 1, context_density)[:, None],
                                  np.linspace(0, 1, context_density)[:, None])

    all_params = cross_product(action_pairs, context_pairs)

    all_results = Parallel(n_jobs=18)(delayed(median_cases)(*all_params[i], i) for i in range(len(all_params)))

    pickle.dump((all_params, all_results), open(base_dir + 'covid_params_results.p', 'wb'))

    X = all_params
    y = all_results
    y_hat = ((-y - np.mean(-y)) / np.std(-y))[:, None]
    pickle.dump((X, y_hat), open(base_dir + 'covid_X_y.p', 'wb'))
