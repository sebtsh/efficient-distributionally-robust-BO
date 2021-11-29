"""
Code from https://github.com/peter-i-frazier/group-testing
"""

import yaml
import os
import numpy as np
from scipy.stats import poisson
import functools

# upper bound on how far the recursion can go in the yaml-depency tree
MAX_DEPTH = 5

# simulation parameters which can be included as yaml-keys but are not required
# they are set to a default value of 0 if not included in the yaml-config
DEFAULT_ZERO_PARAMS = ['initial_E_count',
                       'initial_pre_ID_count',
                       'initial_ID_count',
                       'initial_SyID_mild_count',
                       'initial_SyID_severe_count']

# yaml-keys which share the same key as the simulation parameter, and
# can be copied over one-to-one
COPY_DIRECTLY_YAML_KEYS = ['exposed_infection_p', 'expected_contacts_per_day',
                           'perform_contact_tracing', 'contact_tracing_delay',
                           'cases_isolated_per_contact', 'cases_quarantined_per_contact',
                           'use_asymptomatic_testing', 'contact_trace_testing_frac', 'days_between_tests',
                           'test_population_fraction', 'test_protocol_QFNR', 'test_protocol_QFPR',
                           'initial_ID_prevalence', 'population_size', 'daily_outside_infection_p'] + \
                          DEFAULT_ZERO_PARAMS


@functools.lru_cache(maxsize=128)
def poisson_pmf(max_time, mean_time):
    pmf = list()
    for i in range(max_time):
        pmf.append(poisson.pmf(i, mean_time))
    pmf.append(1 - np.sum(pmf))
    return np.array(pmf)


def poisson_waiting_function(max_time, mean_time):
    return lambda n: np.random.multinomial(n, poisson_pmf(max_time, mean_time))


def binomial_exit_function(p):
    return lambda n: np.random.binomial(n, p)


def subdivide_severity(prob_severity_given_age, prob_infection, prob_age):
    """
    prob_severity_given_age is a matrix where element [i,j] is the probability that someone in age group i has a severity of j
	prob_infection is a vector where element i is the probability of infection given close contact for age group i
	prob_age is the proportion of the population that is in age group i
	The return vector is the probability that an infected patient is of severity j
    :param prob_severity_given_age:
    :param prob_infection:
    :param prob_age:
    :return:
    """

    # Check everything is the right size
    num_age_groups = prob_severity_given_age.shape[0]
    num_severity_levels = prob_severity_given_age.shape[1]

    assert np.all(np.sum(prob_severity_given_age, axis=1) > 0.9999) == True
    assert len(prob_infection) == num_age_groups
    assert len(prob_age) == num_age_groups
    assert np.sum(prob_age) > 0.9999

    S = list()
    for severity_level in range(num_severity_levels):
        total = 0
        for age_group in range(num_age_groups):
            total += prob_severity_given_age[age_group, severity_level] * prob_infection[age_group] * prob_age[
                age_group]
        S.append(total)

    S = S / np.sum(S)

    return S


def update_sev_prevalence(curr_prevalence_dist, new_asymptomatic_pct):
    new_dist = [new_asymptomatic_pct]
    remaining_mass = sum(curr_prevalence_dist[1:])

    # need to scale so that param_val + x * remaning_mass == 1
    scale = (1 - new_asymptomatic_pct) / remaining_mass
    idx = 1
    while idx < len(curr_prevalence_dist):
        new_dist.append(curr_prevalence_dist[idx] * scale)
        idx += 1
    assert (np.isclose(sum(new_dist), 1))
    return np.array(new_dist)


def load_age_sev_params(param_file):
    with open(param_file) as f:
        age_sev_params = yaml.load(f)

    subparams = age_sev_params['prob_severity_given_age']
    prob_severity_given_age = np.array([
        subparams['agegroup1'],
        subparams['agegroup2'],
        subparams['agegroup3'],
        subparams['agegroup4'],
        subparams['agegroup5'],
    ])

    prob_infection = np.array(age_sev_params['prob_infection_by_age'])
    prob_age = np.array(age_sev_params['prob_age'])
    return subdivide_severity(prob_severity_given_age, prob_infection, prob_age)


def load_params(param_file=None, param_file_stack=[], additional_params={}):
    if param_file is not None:
        assert (len(additional_params) == 0)
        with open(param_file) as f:
            params = yaml.load(f)
        # go through params that point to other directories: start by changing
        # the current working directory so that relative file paths can be parsed
        cwd = os.getcwd()

        nwd = os.path.dirname(os.path.realpath(param_file))
        os.chdir(nwd)

    else:
        params = additional_params

    if '_inherit_config' in params:
        if len(param_file_stack) >= MAX_DEPTH:
            raise (Exception("yaml config dependency depth exceeded max depth"))
        new_param_file = params['_inherit_config']
        scenario_name, base_params = load_params(new_param_file, param_file_stack + [param_file])
    else:
        scenario_name = None
        base_params = {}

    if '_age_severity_config' in params:
        age_sev_file = params['_age_severity_config']
        severity_dist = load_age_sev_params(age_sev_file)
        base_params['severity_prevalence'] = severity_dist
    else:
        severity_dist = None

    if '_scenario_name' in params:
        scenario_name = params['_scenario_name']
    else:
        # the top-level param-file needs a name
        if len(param_file_stack) == 0:
            raise (Exception("need to specify a _scenario_name value"))

    if param_file != None:
        # change working-directory back
        os.chdir(cwd)

    # process the main params loaded from yaml, as well as the additional_params
    # optionally passed as an argument, and store them in base_params
    for yaml_key, val in params.items():
        # skip the meta-params
        if yaml_key[0] == '_':
            continue

        if yaml_key == 'ID_time_params':
            assert (len(val) == 2)
            mean_time_ID = val[0]
            max_time_ID = val[1]
            base_params['max_time_ID'] = max_time_ID
            base_params['ID_time_function'] = poisson_waiting_function(max_time_ID, mean_time_ID)

        elif yaml_key == 'E_time_params':
            assert (len(val) == 2)
            base_params['max_time_exposed'] = val[1]
            base_params['exposed_time_function'] = poisson_waiting_function(val[1], val[0])

        elif yaml_key == 'Sy_time_params':
            assert (len(val) == 2)
            base_params['max_time_SyID_mild'] = val[1]
            base_params['SyID_mild_time_function'] = poisson_waiting_function(val[1], val[0])
            base_params['max_time_SyID_severe'] = val[1]
            base_params['SyID_severe_time_function'] = poisson_waiting_function(val[1], val[0])

        elif yaml_key == 'asymptomatic_daily_self_report_p':
            base_params['mild_symptoms_daily_self_report_p'] = val

        elif yaml_key == 'symptomatic_daily_self_report_p':
            base_params['severe_symptoms_daily_self_report_p'] = val

        elif yaml_key == 'daily_leave_QI_p':
            base_params['sample_QI_exit_function'] = binomial_exit_function(
                val)  # (lambda n: np.random.binomial(n, val))

        elif yaml_key == 'daily_leave_QS_p':
            base_params['sample_QS_exit_function'] = binomial_exit_function(
                val)  # (lambda n: np.random.binomial(n, val))

        elif yaml_key == 'asymptomatic_pct_mult':
            if 'severity_prevalence' not in base_params:
                raise (Exception("encountered asymptomatic_pct_mult with no corresponding severity_dist to modify"))
            new_asymptomatic_p = val * base_params['severity_prevalence'][0]
            base_params['severity_prevalence'] = update_sev_prevalence(base_params['severity_prevalence'],
                                                                       new_asymptomatic_p)

        elif yaml_key in COPY_DIRECTLY_YAML_KEYS:
            base_params[yaml_key] = val

        else:
            raise (Exception("encountered unknown parameter {}".format(yaml_key)))

    # the pre-ID state is not being used atm so fill it in with some default params here
    if 'max_time_pre_ID' not in base_params:
        base_params['max_time_pre_ID'] = 4
        base_params['pre_ID_time_function'] = poisson_waiting_function(max_time=4, mean_time=0)

    # the following 'initial_count' variables are all defaulted to 0
    for paramname in DEFAULT_ZERO_PARAMS:
        if paramname not in base_params:
            base_params[paramname] = 0

    if 'pre_ID_state' not in base_params:
        base_params['pre_ID_state'] = 'detectable'

    if 'mild_severity_levels' not in base_params:
        base_params['mild_severity_levels'] = 1

    return scenario_name, base_params
