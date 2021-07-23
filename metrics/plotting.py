# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from matplotlib import cm
from plotly.subplots import make_subplots
from tqdm import trange
from trieste.acquisition import AcquisitionFunction
from trieste.type import TensorType
from trieste.utils import to_numpy
from trieste.utils.pareto import non_dominated

from core.utils import cross_product, adversarial_expectation, get_robust_expectation_and_action


def create_grid(mins: TensorType, maxs: TensorType, grid_density=20):
    """
    Creates a regular 2D grid of size `grid_density^2` between mins and maxs.
    :param mins: list of 2 lower bounds
    :param maxs: list of 2 upper bounds
    :param grid_density: scalar
    :return: Xplot [grid_density**2, 2], xx, yy from meshgrid for the specific formatting of contour / surface plots
    """
    tf.debugging.assert_shapes([(mins, [2]), (maxs, [2])])

    xspaced = np.linspace(mins[0], maxs[0], grid_density)
    yspaced = np.linspace(mins[1], maxs[1], grid_density)
    xx, yy = np.meshgrid(xspaced, yspaced)
    Xplot = np.vstack((xx.flatten(), yy.flatten())).T

    return Xplot, xx, yy


def plot_surface(xx, yy, f, ax, contour=False, alpha=1.0):
    """
    Adds either a contour or a surface to a given ax
    :param xx: input 1, from meshgrid
    :param yy: input2, from meshgrid
    :param f: output, from meshgrid
    :param ax: plt axes object
    :param contour: Boolean
    :param alpha: transparency
    :return:
    """

    if contour:
        return ax.contour(xx, yy, f.reshape(*xx.shape), 80, alpha=alpha)
    else:
        return ax.plot_surface(
            xx,
            yy,
            f.reshape(*xx.shape),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
            alpha=alpha,
        )


def plot_function_1d(obj_func,
                     min_range,
                     max_range,
                     grid_density: int = 1000,
                     title=None,
                     xlabel=None,
                     ylabel=None,
                     figsize=None,
                     dpi=None):
    """
    Plots a 1-D function.
    :param obj_func: a function that returns a n-array given a [n, d] array
    :param min_val:
    :param max_val:
    :param grid_density:
    :param title:
    :param xlabel:
    :param ylabel:
    :return:
    """
    xx = np.expand_dims(np.linspace(min_range, max_range, grid_density), axis=1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(xx, obj_func(xx), label='objective function')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def plot_function_2d(
        obj_func: Callable,
        mins: TensorType,
        maxs: TensorType,
        grid_density: int,
        contour=False,
        log=False,
        title=None,
        xlabel=None,
        ylabel=None,
        figsize=None,
        colorbar=False
):
    """
    2D/3D plot of an obj_func for a grid of size grid_density**2 between mins and maxs
    :param obj_func:
    :param mins: 2 lower bounds
    :param maxs: 2 upper bounds
    :param grid_density: positive integer for the grid size
    :param contour: Boolean. If False, a 3d plot is produced
    :param log: Boolean. If True, the log transformation (log(f - min(f) + 0.1)) is applied
    :param title:
    :param xlabel:
    :param ylabel:
    :param figsize:
    :param colorbar:
    """
    mins = to_numpy(mins)
    maxs = to_numpy(maxs)

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function
    F = to_numpy(obj_func(Xplot))
    if len(F.shape) == 1:
        F = F.reshape(-1, 1)

    n_output = F.shape[1]

    if contour:
        fig, ax = plt.subplots(
            1, n_output, squeeze=False, sharex="all", sharey="all", figsize=figsize
        )
    else:
        fig = plt.figure(figsize=figsize)

    for k in range(F.shape[1]):
        # Apply log transformation
        f = F[:, k]
        if log:
            f = np.log(f - np.min(f) + 1e-1)

        # Either plot contour of surface
        if contour:
            axx = ax[0, k]
        else:
            ax = axx = fig.add_subplot(1, n_output, k + 1, projection="3d")

        plt_obj = plot_surface(xx, yy, f, axx, contour=contour, alpha=1.0)
        if title is not None:
            axx.set_title(title)
        if colorbar:
            fig.colorbar(plt_obj, ax=axx)
        axx.set_xlabel(xlabel)
        axx.set_ylabel(ylabel)
        axx.set_xlim(mins[0], maxs[0])
        axx.set_ylim(mins[1], maxs[1])

    return fig, axx


def plot_acq_function_2d(
        acq_func: AcquisitionFunction,
        mins: TensorType,
        maxs: TensorType,
        grid_density: int = 20,
        contour=False,
        log=False,
        title=None,
        xlabel=None,
        ylabel=None,
        figsize=None,
):
    """
    Wrapper to produce a 2D/3D plot of an acq_func for a grid of size grid_density**2 between mins and maxs
    :param obj_func: a function that returns a n-array given a [n, d] array
    :param mins: 2 lower bounds
    :param maxs: 2 upper bounds
    :param grid_density: positive integer for the grid size
    :param contour: Boolean. If False, a 3d plot is produced
    :param log: Boolean. If True, the log transformation (log(f - min(f) + 0.1)) is applied
    :param title:
    :param xlabel:
    :param ylabel:
    :param figsize:
    """

    def batched_func(x):
        return acq_func(tf.expand_dims(x, axis=-2))

    return plot_function_2d(
        batched_func, mins, maxs, grid_density, contour, log, title, xlabel, ylabel, figsize
    )


def format_point_markers(
        num_pts,
        num_init=None,
        idx_best=None,
        mask_fail=None,
        m_init="x",
        m_add="o",
        c_pass="tab:green",
        c_fail="tab:red",
        c_best="tab:purple",
):
    """
    Prepares point marker styles according to some BO factors
    :param num_pts: total number of BO points
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
    :param m_init: marker for the initial BO points
    :param m_add: marker for the other BO points
    :param c_pass: color for the regular BO points
    :param c_fail: color for the failed BO points
    :param c_best: color for the best BO points
    :return: 2 string vectors col_pts, mark_pts containing marker styles and colors
    """
    if num_init is None:
        num_init = num_pts
    col_pts = np.repeat(c_pass, num_pts)
    col_pts = col_pts.astype("<U15")
    mark_pts = np.repeat(m_init, num_pts)
    mark_pts[num_init:] = m_add
    if mask_fail is not None:
        col_pts[np.where(mask_fail)] = c_fail
    if idx_best is not None:
        col_pts[idx_best] = c_best

    return col_pts, mark_pts


def plot_bo_points_1d(query_points,
                      obj_func,
                      ax,
                      num_init,
                      m_init="x",
                      m_add="o",
                      colormap=cm.rainbow):
    num_add = len(query_points) - num_init
    ax.scatter(query_points[:num_init], obj_func(query_points[:num_init]), c='black', marker=m_init)
    ax.scatter(query_points[num_init:], obj_func(query_points[num_init:]),
               c=colormap(np.arange(0, num_add) / num_add),
               marker=m_add)


def plot_bo_points_2d(query_points,
                      ax,
                      num_init,
                      maximizer,
                      m_init="x",
                      m_add="o",
                      m_max="*",
                      colormap=cm.rainbow):
    num_add = len(query_points) - num_init
    ax.scatter(maximizer[:, 0], maximizer[:, 1], c='black', marker=m_max, zorder=11)
    ax.scatter(query_points[:num_init, 0], query_points[:num_init, 1], c='black', marker=m_init, zorder=11)
    ax.scatter(query_points[num_init:, 0], query_points[num_init:, 1],
               c=colormap(np.arange(0, num_add) / num_add),
               marker=m_add,
               zorder=10)


def plot_bo_points(
        pts,
        ax,
        num_init=None,
        idx_best=None,
        mask_fail=None,
        obs_values=None,
        m_init="x",
        m_add="o",
        c_pass="tab:green",
        c_fail="tab:red",
        c_best="tab:purple",
):
    """
    Adds scatter points to an existing subfigure. Markers and colors are chosen according to BO factors.
    :param pts: [N, 2] x inputs
    :param ax: a plt axes object
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
    :param obs_values: optional [N] outputs (for 3d plots)
    """

    num_pts = pts.shape[0]

    col_pts, mark_pts = format_point_markers(
        num_pts, num_init, idx_best, mask_fail, m_init, m_add, c_pass, c_fail, c_best
    )

    if obs_values is None:
        for i in range(pts.shape[0]):
            ax.scatter(pts[i, 0], pts[i, 1], c=col_pts[i], marker=mark_pts[i])
    else:
        for i in range(pts.shape[0]):
            ax.scatter(pts[i, 0], pts[i, 1], obs_values[i], c=col_pts[i], marker=mark_pts[i])


def plot_mobo_points_in_obj_space(
        obs_values,
        num_init=None,
        mask_fail=None,
        figsize=None,
        xlabel="Obj 1",
        ylabel="Obj 2",
        zlabel="Obj 3",
        title=None,
        m_init="x",
        m_add="o",
        c_pass="tab:green",
        c_fail="tab:red",
        c_pareto="tab:purple",
        only_plot_pareto=False,
):
    """
    Adds scatter points in objective space, used for multi-objective optimization (2 objective only).
    Markers and colors are chosen according to BO factors.
    :param obs_values:
    :param num_init: initial number of BO points
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
    :param title:
    :param xlabel:
    :param ylabel:
    :param figsize:
    :param only_plot_pareto: if set true, only plot the pareto points
    """
    obj_num = obs_values.shape[-1]
    tf.debugging.assert_shapes([])
    assert obj_num == 2 or obj_num == 3, NotImplementedError(
        f"Only support 2/3-objective functions but found: {obj_num}"
    )

    _, dom = non_dominated(obs_values)
    idx_pareto = (
        np.where(dom == 0) if mask_fail is None else np.where(np.logical_and(dom == 0, ~mask_fail))
    )

    pts = obs_values
    num_pts = pts.shape[0]

    col_pts, mark_pts = format_point_markers(
        num_pts, num_init, idx_pareto, mask_fail, m_init, m_add, c_pass, c_fail, c_pareto
    )
    if only_plot_pareto:
        col_pts = col_pts[idx_pareto]
        mark_pts = mark_pts[idx_pareto]
        pts = pts[idx_pareto]

    if obj_num == 2:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for i in range(pts.shape[0]):
        ax.scatter(*pts[i], c=col_pts[i], marker=mark_pts[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if obj_num == 3:
        ax.set_zlabel(zlabel)
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_mobo_history(
        obs_values,
        metric_func,
        num_init=None,
        mask_fail=None,
        figsize=None,
):
    """
    Draw the performance measure for multi-objective optimization
    :param obs_values:
    :param metric_func: a callable function calculate metric score
                        metric = measure_func(observations)
    :param num_init:
    :param mask_fail:
    :param figsize
    """

    fig, ax = plt.subplots(figsize=figsize)
    size, obj_num = obs_values.shape

    if mask_fail is not None:
        obs_values[mask_fail] = [np.inf] * obj_num

    ax.plot([metric_func(obs_values[:pts, :]) for pts in range(size)], color="tab:orange")
    ax.axvline(x=num_init - 0.5, color="tab:blue")
    return fig, ax


def plot_regret(obj_func,
                search_points,
                query_points,
                maximizers=None,
                regret_type='immediate',
                fvals_source='best_seen',
                figsize=None,
                dpi=None):
    """

    :param obj_func:
    :param maximizers:
    :param search_points:
    :param query_points:
    :param regret_type: Either 'immediate', 'average', or cumulative
    :param fvals_source: Either 'gp_belief', 'best_seen', or 'queries'
    :param figsize:
    :param dpi:
    :return:
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    global_max = np.max(obj_func(search_points))

    if fvals_source == 'gp_belief':
        regret = np.squeeze(global_max - obj_func(maximizers))
    elif fvals_source == 'best_seen':
        max_fvals = []
        query_fvals = obj_func(query_points)
        for i in range(len(query_points)):
            max_fvals.append(np.max(query_fvals[:i + 1]))
        regret = global_max - np.array(max_fvals)
    elif fvals_source == 'queries':
        regret = np.squeeze(global_max - obj_func(query_points))
    else:
        raise Exception("Wrong parameters passed to plot_regret")
    if regret_type == 'immediate':
        ax.plot(np.arange(0, len(regret)), regret, marker='x')
    elif regret_type == 'average':
        averages = []
        for i in range(len(maximizers)):
            averages.append(np.mean(regret[:i + 1]))
        ax.plot(np.arange(0, len(averages)), averages, marker='x')
    else:
        raise Exception("Wrong parameters passed to plot_regret")
    ax.set_title(regret_type + ' regret')
    return fig, ax


def plot_robust_regret(obj_func,
                       query_points,
                       action_points,
                       context_points,
                       kernel,
                       ref_dist_func,
                       margin_func,
                       divergence,
                       figsize=None,
                       dpi=None):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    regrets = []
    cumulative_regrets = []
    print("Plotting cumulative robust regret")
    if divergence == 'MMD':
        M = kernel(context_points)
    else:
        M = None
    for t in trange(len(query_points)):
        domain = cross_product(query_points[t:t + 1, 0:1], context_points)
        f = obj_func(domain)
        query_expectation, w = adversarial_expectation(f=f, M=M, w_t=ref_dist_func(t), epsilon=margin_func(t),
                                                       divergence=divergence)
        robust_expectation, robust_action = get_robust_expectation_and_action(action_points=action_points,
                                                                              context_points=context_points,
                                                                              kernel=kernel,
                                                                              fvals_source='obj_func',
                                                                              ref_dist=ref_dist_func(t),
                                                                              divergence=divergence,
                                                                              epsilon=margin_func(t),
                                                                              obj_func=obj_func)
        print("t = {}".format(t))
        print("query_point = {}".format(query_points[t:t + 1, 0:1]))
        print("robust_action = {}".format(robust_action))
        print("robust_expectation = {}".format(robust_expectation))
        print("query_expectation = {}".format(query_expectation))
        #adv_mean = w @ context_points
        #adv_var = w @ ((context_points - adv_mean) ** 2)
        #print("adversarially chosen distribution has mean = {} and variance = {}".format(adv_mean, adv_var))
        print("adversarially chosen distribution: {}".format(w))
        print("===========")
        regrets.append(robust_expectation - query_expectation)
        cumulative_regrets.append(np.sum(regrets))
    ax.plot(np.arange(0, len(regrets)), cumulative_regrets, marker='x')
    ax.set_title('Cumulative robust regret')
    return fig, ax


def plot_gp_1d(model,
               min_range,
               max_range,
               grid_density=1000,
               data=None,
               data_mean=0,
               data_std=1,
               title="",
               xlabel="",
               ylabel="",
               point_size=10):
    """
    Plots a 1-D GP posterior. If the data was standardized, provide the data mean and std for the plot to reflect
    the true values
    :param model: gpflow GP model
    :param min_range:
    :param max_range:
    :param grid_density:
    :param data: array of shape (n, d). Data points to show with posterior
    :param data_mean: float. Mean that was used if data was standardized
    :param data_std: float. Std that was used if data was standardized
    :param title:
    :param xlabel:
    :param ylabel:
    :param point_size:
    :return:
    """
    xx = np.expand_dims(np.linspace(min_range, max_range, grid_density), axis=1)

    fig, ax = plt.subplots()
    Ymean, Yvar = model.predict_y(xx)
    Ymean = Ymean.numpy().squeeze() * data_std + data_mean  # If data was standardized, optionally shift back to orig.
    Ystd = tf.sqrt(Yvar).numpy().squeeze() * data_std
    ax.plot(xx, Ymean, label='posterior')
    for k in (1, 2):
        lb = (Ymean - k * Ystd).squeeze()
        ub = (Ymean + k * Ystd).squeeze()
        ax.fill_between(np.squeeze(xx), lb, ub, color="lightblue", alpha=1 - 0.05 * k ** 3)
    if data is not None:
        X, Y = data
        ax.scatter(X, Y * data_std + data_mean, s=point_size)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def plot_gp_2d(
        model,
        mins: TensorType,
        maxs: TensorType,
        grid_density: int = 20,
        contour=True,
        xlabel=None,
        ylabel=None,
        figsize=None,
        predict_y=False,
):
    """
    2D/3D plot of a gp model for a grid of size grid_density**2 between mins and maxs
    :param model: a gpflow model
    :param mins: 2 lower bounds
    :param maxs: 2 upper bounds
    :param grid_density: positive integer for the grid size
    :param contour: Boolean. If False, a 3d plot is produced
    :param xlabel: optional string
    :param ylabel: optional string
    :param figsize:
    """
    mins = to_numpy(mins)
    maxs = to_numpy(maxs)

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function
    if predict_y:
        Fmean, Fvar = model.predict_y(Xplot)
    else:
        Fmean, Fvar = model.predict_f(Xplot)

    n_output = Fmean.shape[1]
    for i in range(2):
        if contour:
            fig, ax = plt.subplots(
                n_output, 1, squeeze=False, sharex="all", sharey="all", figsize=figsize  # TODO: Change 1 to 2 if want to plot variance later
            )
            ax[0, 0].set_xlim(mins[0], maxs[0])
            ax[0, 0].set_ylim(mins[1], maxs[1])
        else:
            fig = plt.figure(figsize=figsize)

        for k in range(n_output):
            # Apply log transformation
            fmean = Fmean[:, k].numpy()
            fvar = Fvar[:, k].numpy()

            # Either plot contour of surface
            if contour:
                if i == 0:
                    fvals = fmean
                    title = "mean"
                elif i == 1:
                    fvals = fvar
                    title = "variance"
                axx = ax[k, 0]
                plt_obj = plot_surface(xx, yy, fvals, ax[k, 0], contour=contour, alpha=1.0)
                #plot_surface(xx, yy, fvar, ax[k, 1], contour=contour, alpha=1.0)
                ax[k, 0].set_title(title)
                #ax[k, 1].set_title("variance")
                ax[k, 0].set_xlabel(xlabel)
                ax[k, 0].set_ylabel(ylabel)
                #ax[k, 1].set_xlabel(xlabel)
                #ax[k, 1].set_ylabel(ylabel)
                fig.colorbar(plt_obj, ax=axx)
            else:
                ax = axx = fig.add_subplot(1, n_output, k + 1, projection="3d")
                plot_surface(xx, yy, fmean, axx, contour=contour, alpha=0.5)
                ucb = fmean + 2.0 * np.sqrt(fvar)
                lcb = fmean - 2.0 * np.sqrt(fvar)
                plot_surface(xx, yy, ucb, axx, contour=contour, alpha=0.1)
                plot_surface(xx, yy, lcb, axx, contour=contour, alpha=0.1)
                axx.set_xlabel(xlabel)
                axx.set_ylabel(ylabel)
                axx.set_xlim(mins[0], maxs[0])
                axx.set_ylim(mins[1], maxs[1])

    return fig, ax


def format_point_markers(
        num_pts,
        num_init,
        idx_best=None,
        mask_fail=None,
        m_init="x",
        m_add="circle",
        c_pass="green",
        c_fail="red",
        c_best="darkmagenta",
):
    """
    Prepares point marker styles according to some BO factors
    :param num_pts: total number of BO points
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
    :param m_init: marker for the initial BO points
    :param m_add: marker for the other BO points
    :param c_pass: color for the regular BO points
    :param c_fail: color for the failed BO points
    :param c_best: color for the best BO points
    :return: 2 string vectors col_pts, mark_pts containing marker styles and colors
    """

    col_pts = np.repeat(c_pass, num_pts).astype("<U15")
    mark_pts = np.repeat(m_init, num_pts).astype("<U15")
    mark_pts[num_init:] = m_add
    if mask_fail is not None:
        col_pts[mask_fail] = c_fail
    if idx_best is not None:
        col_pts[idx_best] = c_best

    return col_pts, mark_pts


def add_surface_plotly(xx, yy, f, fig, alpha=1.0, figrow=1, figcol=1):
    """
    Adds a surface to an existing plotly subfigure
    :param xx: [n, n] array (input)
    :param yy: [n, n] array (input)
    :param f: [n, n] array (output)
    :param fig: the current plotly figure
    :param alpha: transparency
    :param figrow: row index of the subfigure
    :param figcol: column index of the subfigure
    :return: updated plotly figure
    """

    d = pd.DataFrame(f.reshape([xx.shape[0], yy.shape[1]]), index=xx, columns=yy)

    fig.add_trace(
        go.Surface(z=d, x=xx, y=yy, showscale=False, opacity=alpha, colorscale="viridis"),
        row=figrow,
        col=figcol,
    )
    return fig


def add_bo_points_plotly(x, y, z, fig, num_init, idx_best=None, mask_fail=None, figrow=1, figcol=1):
    """
    Adds scatter points to an existing subfigure. Markers and colors are chosen according to BO factors.
    :param x: [N] x inputs
    :param y: [N] y inputs
    :param z: [N] z outputs
    :param fig: the current plotly figure
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
    :param figrow: row index of the subfigure
    :param figcol: column index of the subfigure
    :return: a plotly figure
    """
    num_pts = x.shape[0]

    col_pts, mark_pts = format_point_markers(num_pts, num_init, idx_best, mask_fail)

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=4, color=col_pts, symbol=mark_pts, opacity=0.8),
        ),
        row=figrow,
        col=figcol,
    )

    return fig


def plot_gp_plotly(model, mins: TensorType, maxs: TensorType, grid_density=20):
    """

    :param model: a gpflow model
    :param mins: list of 2 lower bounds
    :param maxs: list of 2 upper bounds
    :param grid_density: integer (grid size)
    :return: a plotly figure
    """
    mins = to_numpy(mins)
    maxs = to_numpy(maxs)

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function
    Fmean, Fvar = model.predict_f(Xplot)

    n_output = Fmean.shape[1]

    fig = make_subplots(
        rows=1, cols=n_output, specs=[np.repeat({"type": "surface"}, n_output).tolist()]
    )

    for k in range(n_output):
        fmean = Fmean[:, k].numpy()
        fvar = Fvar[:, k].numpy()

        lcb = fmean - 2 * np.sqrt(fvar)
        ucb = fmean + 2 * np.sqrt(fvar)

        fig = add_surface_plotly(xx, yy, fmean, fig, alpha=1.0, figrow=1, figcol=k + 1)
        fig = add_surface_plotly(xx, yy, lcb, fig, alpha=0.5, figrow=1, figcol=k + 1)
        fig = add_surface_plotly(xx, yy, ucb, fig, alpha=0.5, figrow=1, figcol=k + 1)

    return fig


def plot_function_plotly(
        obj_func,
        mins: TensorType,
        maxs: TensorType,
        grid_density=20,
        title=None,
        xlabel=None,
        ylabel=None,
        alpha=1.0,
):
    """
    Draws an objective function.
    :obj_func: the vectorized objective function
    :param mins: list of 2 lower bounds
    :param maxs: list of 2 upper bounds
    :param grid_density: integer (grid size)
    :return: a plotly figure
    """

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function
    F = to_numpy(obj_func(Xplot))
    if len(F.shape) == 1:
        F = F.reshape(-1, 1)
    n_output = F.shape[1]

    fig = make_subplots(
        rows=1,
        cols=n_output,
        specs=[np.repeat({"type": "surface"}, n_output).tolist()],
        subplot_titles=title,
    )

    for k in range(n_output):
        f = F[:, k]
        fig = add_surface_plotly(xx, yy, f, fig, alpha=alpha, figrow=1, figcol=k + 1)
        fig.update_xaxes(title_text=xlabel, row=1, col=k + 1)
        fig.update_yaxes(title_text=ylabel, row=1, col=k + 1)

    return fig
