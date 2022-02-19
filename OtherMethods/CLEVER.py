"""
The codes in this file are adapted from adversarial-robustness-toolbox written by Tusted-AI in IBM.
URL: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/metrics/metrics.py
The Adversarial Robustness Toolbox is an open source repository in GitHub and is under the MIT license.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from functools import reduce
import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
import numpy.linalg as la
from scipy.optimize import fmin as scipy_optimizer
from scipy.stats import weibull_min
from tqdm import tqdm

from art.config import ART_NUMPY_DTYPE
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.utils import random_sphere

if TYPE_CHECKING:
    from art.attacks import EvasionAttack
    from art.estimators.classification.classifier import Classifier, ClassifierGradients

logger = logging.getLogger(__name__)

SUPPORTED_METHODS: Dict[str, Dict[str, Any]] = {
    "fgsm": {
        "class": FastGradientMethod,
        "params": {"eps_step": 0.1, "eps_max": 1.0, "clip_min": 0.0, "clip_max": 1.0},
    },
    "hsj": {"class": HopSkipJump, "params": {"max_iter": 50, "max_eval": 10000, "init_eval": 100, "init_size": 100,},},
}


def clever_u(
    classifier: "ClassifierGradients",
    x: np.ndarray,
    nb_batches: int,
    batch_size: int,
    radius: float,
    norm: int,
    c_init: float = 1.0,
    pool_factor: int = 10,
) -> float:
    """
    Compute CLEVER score for an untargeted attack.

    | Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :param x: One input sample.
    :param nb_batches: Number of repetitions of the estimate.
    :param batch_size: Number of random examples to sample per batch.
    :param radius: Radius of the maximum perturbation.
    :param norm: Current support: 1, 2, np.inf.
    :param c_init: initialization of Weibull distribution.
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s.
    :return: CLEVER score.
    """
    # Get a list of untargeted classes
    y_pred = classifier.predict(np.array([x]))
    pred_class = np.argmax(y_pred, axis=1)[0]
    untarget_classes = [i for i in range(classifier.nb_classes) if i != pred_class]

    # Compute CLEVER score for each untargeted class
    score_list = []
    loc_list = []
    for j in tqdm(untarget_classes, desc="CLEVER untargeted"):
        score, loc = clever_t(classifier, x, j, nb_batches, batch_size, radius, norm, c_init, pool_factor)
        score_list.append(score)
        loc_list.append(abs(loc))

    return np.min(score_list), loc_list


def clever_t(
    classifier: "ClassifierGradients",
    x: np.ndarray,
    target_class: int,
    nb_batches: int,
    batch_size: int,
    radius: float,
    norm: int,
    c_init: float = 1.0,
    pool_factor: int = 10,
) -> float:
    """
    Compute CLEVER score for a targeted attack.

    | Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :param x: One input sample.
    :param target_class: Targeted class.
    :param nb_batches: Number of repetitions of the estimate.
    :param batch_size: Number of random examples to sample per batch.
    :param radius: Radius of the maximum perturbation.
    :param norm: Current support: 1, 2, np.inf.
    :param c_init: Initialization of Weibull distribution.
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s.
    :return: CLEVER score.
    """
    # Check if the targeted class is different from the predicted class
    y_pred = classifier.predict(np.array([x]))
    pred_class = np.argmax(y_pred, axis=1)[0]
    if target_class == pred_class:
        raise ValueError("The targeted class is the predicted class.")

    # Check if pool_factor is smaller than 1
    if pool_factor < 1:
        raise ValueError("The `pool_factor` must be larger than 1.")

    # Some auxiliary vars
    grad_norm_set = []
    dim = reduce(lambda x_, y: x_ * y, x.shape, 1)
    shape = [pool_factor * batch_size]
    shape.extend(x.shape)

    # Generate a pool of samples
    rand_pool = np.reshape(
        random_sphere(nb_points=pool_factor * batch_size, nb_dims=dim, radius=radius, norm=norm), shape,
    )
    rand_pool += np.repeat(np.array([x]), pool_factor * batch_size, 0)
    rand_pool = rand_pool.astype(ART_NUMPY_DTYPE)
    if hasattr(classifier, "clip_values") and classifier.clip_values is not None:
        np.clip(rand_pool, classifier.clip_values[0], classifier.clip_values[1], out=rand_pool)

    # Change norm since q = p / (p-1)
    if norm == 1:
        norm = np.inf
    elif norm == np.inf:
        norm = 1
    elif norm != 2:
        raise ValueError("Norm {} not supported".format(norm))

    # Loop over the batches
    for _ in range(nb_batches):
        # Random generation of data points
        sample_xs = rand_pool[np.random.choice(pool_factor * batch_size, batch_size)]

        # Compute gradients
        grads = classifier.class_gradient(sample_xs)
        if np.isnan(grads).any():
            raise Exception("The classifier results NaN gradients.")

        grad = grads[:, pred_class] - grads[:, target_class]
        grad = np.reshape(grad, (batch_size, -1))
        grad_norm = np.max(np.linalg.norm(grad, ord=norm, axis=1))
        grad_norm_set.append(grad_norm)

    # Maximum likelihood estimation for max gradient norms
    [_, loc, _] = weibull_min.fit(-np.array(grad_norm_set), c_init, optimizer=scipy_optimizer)

    # Compute function value
    values = classifier.predict(np.array([x]))
    value = values[:, pred_class] - values[:, target_class]

    # Compute scores
    score = np.min([-value[0] / loc, radius])

    return score, loc


def clever_modified(
    classifier: "ClassifierGradients",
    x: np.ndarray,
    nb_batches: int,
    batch_size: int,
    radius: float,
    norm: int,
    c_init: float = 1.0,
    pool_factor: int = 10,
    outputIndex:int = 0
) -> float:
    """
    Compute CLEVER score for a targeted attack.

    | Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :param x: One input sample.
    :param target_class: Targeted class.
    :param nb_batches: Number of repetitions of the estimate.
    :param batch_size: Number of random examples to sample per batch.
    :param radius: Radius of the maximum perturbation.
    :param norm: Current support: 1, 2, np.inf.
    :param c_init: Initialization of Weibull distribution.
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s.
    :return: CLEVER score.
    """
    # Check if the targeted class is different from the predicted class
    # y_pred = classifier.predict(np.array([x]))
    # pred_class = np.argmax(y_pred, axis=1)[0]
    pred_class = outputIndex

    # Check if pool_factor is smaller than 1
    if pool_factor < 1:
        raise ValueError("The `pool_factor` must be larger than 1.")

    # Some auxiliary vars
    grad_norm_set = []
    dim = reduce(lambda x_, y: x_ * y, x.shape, 1)
    shape = [pool_factor * batch_size]
    shape.extend(x.shape)

    # Generate a pool of samples
    rand_pool = np.reshape(
        random_sphere(nb_points=pool_factor * batch_size, nb_dims=dim, radius=radius, norm=norm), shape,
    )
    rand_pool += np.repeat(np.array([x]), pool_factor * batch_size, 0)
    rand_pool = rand_pool.astype(ART_NUMPY_DTYPE)
    if hasattr(classifier, "clip_values") and classifier.clip_values is not None:
        np.clip(rand_pool, classifier.clip_values[0], classifier.clip_values[1], out=rand_pool)

    # Change norm since q = p / (p-1)
    if norm == 1:
        norm = np.inf
    elif norm == np.inf:
        norm = 1
    elif norm != 2:
        raise ValueError("Norm {} not supported".format(norm))

    # Loop over the batches
    for _ in range(nb_batches):
        # Random generation of data points
        sample_xs = rand_pool[np.random.choice(pool_factor * batch_size, batch_size)]

        # Compute gradients
        grads = classifier.class_gradient(sample_xs)
        if np.isnan(grads).any():
            raise Exception("The classifier results NaN gradients.")

        grad = grads[:, pred_class]
        grad = np.reshape(grad, (batch_size, -1))
        grad_norm = np.max(np.linalg.norm(grad, ord=norm, axis=1))
        grad_norm_set.append(grad_norm)

    # Maximum likelihood estimation for max gradient norms
    [_, loc, _] = weibull_min.fit(-np.array(grad_norm_set), c_init, optimizer=scipy_optimizer)

    return abs(loc)