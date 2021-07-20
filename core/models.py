from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import gpflow as gpf
from gpflow.models import GPModel
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.utilities.traversal import _get_leaf_components, deepcopy
from trieste.data import Dataset
from trieste.type import TensorType


class ModelOptModule(ABC):
    """
    Abstract class that packages a GP model, data, and optimization routine together.
    """

    def __init__(self):
        self.noise_variance = None
        self._dataset = None
        self._gp = None
        self._optimizer = None

    @property
    def dataset(self):
        return self._dataset

    @property
    def gp(self):
        return self._gp

    @property
    def optimizer(self):
        return self._optimizer

    @dataset.setter
    def dataset(self, dataset: Dataset):
        self._dataset = dataset

    @gp.setter
    def gp(self, gp: GPModel):
        self._gp = gp

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @abstractmethod
    def update_dataset(self, dataset: Dataset):
        """
        Updates the dataset in the module and the GP.
        :param data: Trieste Dataset
        :return:
        """
        pass


    @abstractmethod
    def optimize(self):
        """
        Optimizes the GP hyperparameters.
        :return:
        """
        pass

    @abstractmethod
    def predict_f(self, data: TensorType) -> Tuple[TensorType, TensorType]:
        """
        Returns the posterior mean and variance over the domain for the main function f. For single output GPs, this
        should be the same function as in the GPflow models. For heteroscedastic GPs or other multi-output GPs,
        be careful to only output the main f values here, and use other functions for the other outputs.
        :param data: Array of shape (m, d)
        :return: tuple with 2 arrays of shape (m, 1)
        """
        pass

    @abstractmethod
    def predict_y(self, data: TensorType) -> Tuple[TensorType, TensorType]:
        """
        Returns the posterior distribution over observations. Includes the observational variance.
        :param data: Array of shape (m, d)
        :return: tuple with 2 arrays of shape (m, 1)
        """
        pass

    def get_params(self) -> dict:
        return _get_leaf_components(self.gp)


class GPRModule(ModelOptModule):
    """
    Basic GP regression model.
    """

    def __init__(self,
                 dims,
                 kernel: Kernel,
                 noise_variance: float,
                 dataset: Optional[Dataset] = None,
                 mean_func: Optional[MeanFunction] = None,
                 opt_max_iter: Optional[int] = 100):
        super().__init__()
        self.dataset = dataset
        self.dims = dims
        self.noise_variance = noise_variance
        if dataset is not None:
            self.gp = gpf.models.GPR(self.dataset.astuple(),
                                     kernel,
                                     mean_function=mean_func,
                                     noise_variance=noise_variance)
        else:
            self.gp = gpf.models.GPR((np.array([[np.infty] * dims]), np.array([[np.infty]])),  # placeholder dataset)
                                     kernel,
                                     mean_function=mean_func,
                                     noise_variance=noise_variance)
        self.default_gp = deepcopy(self.gp)  # Copy the initialization so we can reset it easily later
        self.optimizer = gpf.optimizers.Scipy()
        self.opt_max_iter = opt_max_iter

    def update_dataset(self, dataset: Dataset):
        self.dataset = dataset
        self.gp.data = self.dataset.astuple()

    def optimize(self):
        self.gp = deepcopy(self.default_gp)  # Reset to default parameters before optimization
        self.gp.data = self.dataset.astuple()
        self.optimizer.minimize(self.gp.training_loss,
                                self.gp.trainable_variables,
                                options=dict(maxiter=self.opt_max_iter))

    def predict_f(self, data: TensorType) -> Tuple[TensorType, TensorType]:
        return self.gp.predict_f(data)

    def predict_y(self, data: TensorType) -> Tuple[TensorType, TensorType]:
        return self.gp.predict_y(data)
