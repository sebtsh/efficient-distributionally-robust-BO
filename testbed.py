import gpflow as gpf
import numpy as np
import matplotlib.pyplot as plt

from core.models import GPRModule
from core.objectives import get_obj_func
from core.observers import mk_noisy_observer
from core.utils import construct_grid_1d, cross_product, get_discrete_normal_dist_1d, get_discrete_uniform_dist_1d, \
    get_margin, adversarial_expectation, cholesky_inverse, MMD, TV, modified_chi_squared, worst_case_sens, \
    get_cubic_approx_func
from metrics.plotting import plot_function_2d
from tqdm import trange
from pathlib import Path

Path("plots").mkdir(parents=True, exist_ok=True)

acq_name = 'GP-UCB'
obj_func_name = 'rand_func'
divergence = 'TV'  # 'MMD', 'TV' or 'modified_chi_squared'
dims = 2
lowers = [0] * dims
uppers = [1] * dims
grid_density_per_dim = 20
rand_func_num_points = 100
ls = 0.1
obs_variance = 0.001
is_optimizing_gp = False
opt_max_iter = 10
num_bo_iters = 200
num_init_points = 10
beta_const = 0
ref_mean = 0.5
ref_var = 0.05
true_mean = 0.2
true_var = 0.05
seed = 0
show_plots = False

np.random.seed(seed)

f_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls] * dims)
if divergence == 'MMD':
    mmd_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls])  # 1d for now
else:
    mmd_kernel = None

# Get objective function
obj_func = get_obj_func(obj_func_name, lowers, uppers, f_kernel, rand_func_num_points, seed)

# Action space (1d for now)
action_points = construct_grid_1d(lowers[0], uppers[0], grid_density_per_dim)
# Context space (1d for now)
context_points = construct_grid_1d(lowers[1], uppers[1], grid_density_per_dim)
search_points = cross_product(action_points, context_points)

observer = mk_noisy_observer(obj_func, obs_variance)
init_dataset = observer(search_points[np.random.randint(0, len(search_points), num_init_points)])

# Model
model = GPRModule(dims=dims,
                  kernel=f_kernel,
                  noise_variance=obs_variance,
                  dataset=init_dataset,
                  opt_max_iter=opt_max_iter)

# Distribution generating functions
ref_dist_func = lambda x: get_discrete_normal_dist_1d(context_points, ref_mean, ref_var)
# true_dist_func = lambda x: get_discrete_normal_dist_1d(context_points, true_mean, true_var)
true_dist_func = lambda x: get_discrete_uniform_dist_1d(context_points)
margin = get_margin(ref_dist_func(0), true_dist_func(0), mmd_kernel, context_points, divergence)
margin_func = lambda x: margin  # Constant margin for now

ref_dist = ref_dist_func(0)
kernel = mmd_kernel
epsilon = margin_func(0)

num_context_points = len(context_points)
num_action_points = len(action_points)

###################################################################

# Compare DRBO and worst-case sensitivity choices
epsilons = np.linspace(0, 3, 200)  # MMD: 1.2,  # chisquared: 10, TV: 3

for seed in range(10):
    obj_func = get_obj_func(obj_func_name, lowers, uppers, f_kernel, rand_func_num_points, seed)

    drbo_gen_values = []
    worst_case_sens_values = []
    cubic_approx_values = []

    for eps in trange(len(epsilons)):
        epsilon = epsilons[eps]
        # DRBO-General
        adv_expectations = []
        # Worst-case sensitivity
        adv_lower_bounds = []
        # Cubic approximation of adversarial expectation
        cubic_values = []

        for i in range(num_action_points):
            tiled_action = np.tile(action_points[i:i + 1], (num_context_points, 1))  # (num_context_points, d_x)
            action_contexts = np.concatenate([tiled_action, context_points],
                                             axis=-1)  # (num_context_points, d_x + d_c)
            fvals = np.squeeze(obj_func(action_contexts))
            if divergence == 'MMD':
                M = kernel(context_points)
            else:
                M = None

            # DRBO-General
            expectation, _ = adversarial_expectation(f=fvals,
                                                     M=M,
                                                     w_t=ref_dist,
                                                     epsilon=epsilon,
                                                     divergence=divergence)

            adv_expectations.append(expectation)

            # Worst-case sensitivity
            expected_fvals = np.sum(ref_dist * fvals)  # SAA
            worst_case_sensitivity = worst_case_sens(fvals=fvals,
                                                     context_points=context_points,
                                                     kernel=kernel,
                                                     divergence=divergence)

            if divergence == 'MMD':
                sens_factor = epsilon  # might be square root epsilon for others
            elif divergence == 'TV':
                sens_factor = epsilon
            elif divergence == 'modified_chi_squared':
                sens_factor = np.sqrt(epsilon)
            else:
                raise Exception("Invalid divergence")

            adv_lower_bound = expected_fvals - (sens_factor * worst_case_sensitivity)
            adv_lower_bounds.append(adv_lower_bound)

            # Cubic approximation
            V_approx_func = get_cubic_approx_func(context_points,
                                                  fvals,
                                                  kernel,
                                                  ref_dist,
                                                  worst_case_sensitivity,
                                                  divergence)
            cubic_values.append(V_approx_func(epsilon))

        adv_expectations = np.squeeze(np.array(adv_expectations))
        adv_lower_bounds = np.squeeze(np.array(adv_lower_bounds))
        cubic_values = np.squeeze(np.array(cubic_values))

        # Get each algorithm's selected action and the true value
        drbo_gen_val = np.max(adv_expectations)
        worst_case_selection = np.argmax(adv_lower_bounds)
        worst_case_val = adv_expectations[worst_case_selection]
        cubic_selection = np.argmax(cubic_values)
        cubic_val = adv_expectations[cubic_selection]

        drbo_gen_values.append(drbo_gen_val)
        worst_case_sens_values.append(worst_case_val)
        cubic_approx_values.append(cubic_val)

    plt.figure()
    plt.plot(epsilons, drbo_gen_values, label='DRBO-General', color='#d7263d')
    plt.plot(epsilons, worst_case_sens_values, label='Worst-case sens', color='#fbb13c')
    plt.plot(epsilons, cubic_approx_values, label='Cubic approx.', color='#00a6ed')
    plt.legend()
    title = "{}-seed{}-improved".format(divergence, seed)
    plt.title(title)
    plt.xlabel("$\epsilon$")
    plt.ylabel("Adv. expectation of selected action")
    plt.savefig("plots/" + title + ".png")

plt.show()


        # print("Seed: {}".format(seed))
        # print("Adversarial expectations:")
        # print(adv_expectations)
        # print("Adversarial lower bounds:")
        # print(adv_lower_bounds)
        # print("Worst case sensitivities:")
        # print(worstcase_sensitivities)
        # print("Robust action: {}".format(np.argmax(adv_expectations)))
        # print("Worst case sens action: {}".format(np.argmax(adv_lower_bounds)))
        # worstcase_selection = np.argmax(adv_lower_bounds)
        # selection_val = adv_expectations[worstcase_selection]
        # order = np.where(sorted(adv_expectations) == selection_val)[0][0]
        # print("Worst case sens chose position {} best action".format(num_context_points - order))
        # print("Diff in expectation value: {}".format(np.max(adv_expectations) - selection_val))
        # print("==========")
#
###################################################################

# # Plot adversarial expectation as a function of epsilon
# seed = 10
# obj_func = get_obj_func(obj_func_name, lowers, uppers, f_kernel, rand_func_num_points, seed)
# epsilons = np.linspace(0, 1, 2000)
# actions = [2, 7]
# adv_expectations = []
# all_deltas = []
# for i in actions:
#     tiled_action = np.tile(action_points[i:i + 1], (num_context_points, 1))  # (num_context_points, d_x)
#     action_contexts = np.concatenate([tiled_action, context_points],
#                                      axis=-1)  # (num_context_points, d_x + d_c)
#     fvals = np.squeeze(obj_func(action_contexts))
#     M = kernel(context_points)
#     exps = []
#     for j in trange(len(epsilons)):
#         epsilon = epsilons[j]
#         expectation, _ = adversarial_expectation(f=fvals,
#                                                  M=M,
#                                                  w_t=ref_dist,
#                                                  epsilon=epsilon,
#                                                  divergence=divergence)
#
#         exps.append(expectation)
#     adv_expectations.append(exps)
#
#     deltas = []
#     for j in range(len(epsilons) - 1):
#         delta = abs(exps[j + 1] - exps[j])
#         deltas.append(delta)
#     all_deltas.append(deltas)
#
# plt.figure()
# plt.title("Adversarial exp, seed {}".format(seed))
# for i in range(len(actions)):
#     action = actions[i]
#     tiled_action = np.tile(action_points[action:action + 1], (num_context_points, 1))  # (num_context_points, d_x)
#     action_contexts = np.concatenate([tiled_action, context_points],
#                                      axis=-1)  # (num_context_points, d_x + d_c)
#     fvals = np.squeeze(obj_func(action_contexts))
#     f_approx = get_cubic_approx_func(action_points[action:action + 1],
#                                context_points,
#                                obj_func,
#                                kernel,
#                                ref_dist,
#                                worst_case_sens(fvals, context_points, kernel))
#     plt.plot(epsilons, adv_expectations[i], label=action)
#     plt.plot(epsilons, f_approx(epsilons), label="fake {}".format(action))
# plt.legend()
#
# plt.figure()
# plt.title("Deltas, seed {}".format(seed))
# for i in range(len(actions)):
#     plt.plot(epsilons[:-1], all_deltas[i], label=actions[i])
# plt.legend()
#
# plt.show()

###################################################################
