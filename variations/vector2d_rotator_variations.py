# create the parent Rotator class
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from abc import ABC, abstractmethod
import copy
import math

"""
Each Rotator is a functor meaning it is callable after you instantiate it.
Optionally cache previous results by instantiating as
Rotator(cache=True), default is False
example usage:
	rot2d = CORDIC1959()
	result = rot2d(3.14,[0,1])
	print(result)
"""


class Rotator(ABC):

    def __init__(self, cache=False):
        """instantiate Rotator object and create a cache of rotation results

        Args:
            cache (bool, optional): cache previous calls and avoid some duplicate computation. Defaults to False.
        """
        self.__keep_cache = False
        self.callcache = dict()

    def __scale_err(self, theta: float, x: np.array) -> float:
        """find the scale error of this rotator on this input vector and theta
        ideally, rotation does not scale a vector

        Args:
            theta (float): rotation angle in radians
            x (np.array): input 2d vector

        Returns:
            float: percent scale error
        """
        rotnorm = np.linalg.norm(self(theta, x))
        xnorm = np.linalg.norm(x)
        err = xnorm - rotnorm
        if any([i < 0.00001 for i in [err, xnorm, rotnorm]]):
            return 0
        return 100 * abs(abs(xnorm - rotnorm) / xnorm)

    def __angle_error(self, other, theta: float, x: np.array) -> float:
        """find the angle difference between this Rotator and other Rotator on this input vector and theta

        Args:
            other (Rotator): reference rotation
            theta (float): rotation angle in radians
            x (np.array): input 2d vector

        Returns:
            float: rotation error in degrees
        """
        refrot = other(theta, x)
        thisrot = self(theta, x)
        refang = np.arctan2(*refrot)
        thisang = np.arctan2(*thisrot)
        angle_diff = abs(thisang) - abs(refang)
        return (
            (angle_diff - (2 * np.pi) if angle_diff > np.pi else angle_diff)
            * 180
            / np.pi
        )

    def compare_angle_error(self, other, x1s=0, x2s=0, mesh=0) -> float:
        """runs a large testset of rotation angles and vectors on both self (Rotator) and other (Rotator)
        produces and shows a 3d surface plot of input vector vs maximum rotation error (testing -45 to 45 degrees)
        returns the maximum rotation angle difference between the reference Rotator (other) and self over the whole testset
        Also allows for specifying custom test set with x1s, x2s, or mesh options

        Args:
            other (Rotator): Rotator which is used as a reference when doing angle error comparison
            x1s (int, optional): list of first input dimension to attempt. Defaults to the built in test.
            x2s (int, optional): list of second input dimension to add to the testset. Defaults to the built in test.
            mesh (int, optional): directly specfying the testset as a 2d mesh matrix. Defaults to the built in test.

        Returns:
            float: Maxmium degrees angle difference between reference rotation and this rotation for all elements in the testset
        """
        thetas = np.arange(-np.pi / 4, np.pi / 4, 0.04, dtype=float)
        x1s = (
            np.arange(-128, 127, 6, dtype=float)
            if not isinstance(x1s, Iterable)
            else x1s
        )
        x2s = (
            np.arange(-128, 127, 6, dtype=float)
            if not isinstance(x2s, Iterable)
            else x2s
        )
        X1s, X2s = np.meshgrid(x1s, x2s) if not isinstance(mesh, Iterable) else mesh
        errs = np.zeros_like(X1s)
        for i in range(X1s.shape[0]):
            for j in range(X1s.shape[1]):
                # compute angle which produces largest deviation from reference
                errs[i, j] = max(
                    [
                        self.__angle_error(other, theta, [X1s[i, j], X2s[i, j]])
                        for theta in thetas
                    ]
                )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X1s, X2s, errs)
        plt.show()
        return errs.max()

    def scale_error(self, x1s=0, x2s=0, mesh=0):
        # test VALID angles (-45 to 45 degrees) and input vectors
        thetas = np.arange(-np.pi / 4, np.pi / 4, 0.04, dtype=float)
        x1s = (
            np.arange(-128, 127, 6, dtype=float)
            if not isinstance(x1s, Iterable)
            else x1s
        )
        x2s = (
            np.arange(-128, 127, 6, dtype=float)
            if not isinstance(x2s, Iterable)
            else x2s
        )
        X1s, X2s = np.meshgrid(x1s, x2s) if not isinstance(mesh, Iterable) else mesh
        errs = np.zeros_like(X1s)
        for i in range(X1s.shape[0]):
            for j in range(X1s.shape[1]):
                # compute angle which produces largest deviation from reference
                errs[i, j] = max(
                    [
                        self.__scale_err(theta, [X1s[i, j], X2s[i, j]])
                        for theta in thetas
                    ]
                )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X1s, X2s, errs)
        ax.set_zlim3d(0, 100)
        plt.show()
        return errs.max()

    def trivial_rotator(cls, theta: float, x: np.array):
        """trivial rotator which takes any real angle and outputs an angle -45 to 45 and a rotated vector

        Args:
            theta (float): any real angle in radians
            x (np.array): input 2d vector

        Returns:
            float, np.array: theta between -pi/4 to pi/4 and np.array
        """
        # copy x so it is not modified
        x = copy.deepcopy(x)
        # place theta within -pi to pi
        theta = theta % (2 * np.pi) if theta > 0 else theta % (-2 * np.pi)
        if abs(theta) > np.pi:
            theta = theta - np.pi if theta > 0 else theta + np.pi
        # trivial rotation
        theta_gt_piby4 = abs(theta) > (np.pi / 4)
        theta_gt_3piby4 = abs(theta) > (3 * np.pi / 4)
        negate_x0 = theta_gt_piby4 and (theta_gt_3piby4 or theta > 0)
        negate_x1 = theta_gt_piby4 and (theta_gt_3piby4 or theta <= 0)
        swap = theta_gt_piby4 and (not theta_gt_3piby4)
        x[0] = -x[0] if negate_x0 else x[0]
        x[1] = -x[1] if negate_x1 else x[1]
        if swap:
            temp = x[0]
            x[0] = x[1]
            x[1] = temp
        # update theta
        rotmag = np.pi if theta_gt_3piby4 else (np.pi / 2 if theta_gt_piby4 else 0)
        rtheta = theta + rotmag if theta < 0 else theta - rotmag
        return rtheta, x

    def __call__(self, theta: float, x: np.array) -> np.array:
        """Rotate vector x by angle theta.
        NOTE: this function will not modify x

        Args:
            theta (float): target rotation angle in radians
            x (np.array): 2d input vector

        Returns:
            np.array: 2d output rotation
        """
        theta, x = self.trivial_rotator(theta, x)
        # cache lookup or direct compute
        if not self.__keep_cache:
            return self.compute_rotation(theta, x)
        pt = (x[0], x[1])
        pt_thetacache = self.callcache.get(pt, False)
        if pt_thetacache is not False:
            result = pt_thetacache.get(theta, False)
            if result is not False:
                return result
            result = self.compute_rotation(theta, x)
            pt_thetacache[theta] = result
            return result
        self.callcache[pt] = dict()
        result = self.compute_rotation(theta, x)
        self.callcache[pt][theta] = result
        return result

    @abstractmethod
    def compute_rotation(self, theta: float, x: np.array) -> np.array:
        """Rotate vector x by angle theta

        Args:
            theta (float): target rotation angle in radians
            x (np.array): 2d input vector

        Returns:
            np.array: 2d output rotation
        """
        return


class PerfectRotator(Rotator):
    """0 error rotator"""

    def compute_rotation(self, theta: float, x: np.array) -> np.array:
        sin_act = np.sin(theta)
        cos_act = np.cos(theta)
        # Real Rotation Matrix
        rotation_matrix = np.array([[cos_act, -sin_act], [sin_act, cos_act]])
        # Rotate vector x by angle theta using rotation matrix
        y_act = rotation_matrix @ x
        return y_act


class CORDIC1959(Rotator):
    """1959 CORDIC rotator
    self.NITER: a postive integer determining number of stages
    """

    def __init__(self, NITER: int = 5):
        super().__init__()
        self.NITER = 5

    @property
    def NITER(self) -> int:
        return self.__NITER

    @NITER.setter
    def NITER(self, value: int):
        ival = int(value)
        if not isinstance(ival, int) or ival < 0:
            raise ValueError("value can only be set to positive int")
        self.__NITER = ival
        self.callcache.clear()

    def compute_rotation(self, theta: float, x: np.array) -> np.array:
        vec = copy.deepcopy(x)
        # precompute thetas
        pcthetas = list()
        for tanexp in range(self.NITER):
            pcthetas.append(math.atan(2 ** (-tanexp)))
        # precompute cos correction factor
        coscorrection = float("1")
        for pctheta in pcthetas:
            coscorrection *= np.cos(pctheta)
        # init values and perform n rotations
        current_rotation = float("0")
        for tanexp, pctheta in enumerate(pcthetas):
            tantheta = 2 ** (-tanexp)
            startvec = copy.deepcopy(vec)
            if current_rotation > theta:
                vec[0] = startvec[0] + (tantheta * startvec[1])
                vec[1] = startvec[1] - (tantheta * startvec[0])
                current_rotation -= pctheta
            else:
                vec[0] = startvec[0] - (tantheta * startvec[1])
                vec[1] = startvec[1] + (tantheta * startvec[0])
                current_rotation += pctheta
        return [coscorrection * ele for ele in vec]


class FOE(Rotator):
    """First Order Estimate rotator
    estimates sin and cos using linear piecewise functions
    """

    def compute_rotation(self, theta: float, x: np.array) -> float:
        # Estimate sin and cos values
        sin_est = theta
        cos_est = 1 - np.sign(theta) * theta / 4
        # Estimate rotation using estimated sin and cos values
        rotation_matrix_est = np.array([[cos_est, -sin_est], [sin_est, cos_est]])
        return rotation_matrix_est @ x


class DoubleFOE(Rotator):
    """Double First Order Estimate rotator
    estimates sin and cos using linear piecewise functions
    estimates second dim of rotation from the first dim
    """

    def compute_rotation(self, theta: float, x: np.array) -> np.array:
        sign = lambda i: 1 if i > 0 else -1
        cos_est = lambda ang: 1 - sign(ang) * ang / 4
        temp = x[0] + sign(theta) * x[1] / 4
        z2 = temp * theta + x[1]
        z1 = (temp + sign(theta) * x[1] / 8) - sign(theta) * (z2 / 4 + z2 / 8)
        return np.array([z1, z2])


class DoubleFOE_Advanced(Rotator):
    """Advanced First Order Estimate rotator
    estimates sin and cos using linear piecewise functions
    estimates second dim of rotation from the first dim
    uses more complicated piecewise functions to perform estimates
    """

    def compute_rotation(self, theta: float, x: np.array) -> np.array:
        sign = lambda i: 1 if i > 0 else -1
        cos_est = (
            lambda ang: 1
            - sign(ang) * ang / 4
            + (-sign(ang) * ang / 16 if abs(ang) > 0.6 else 0)
        )
        pseudonorm = (
            lambda ang, xin: xin[0]
            + sign(ang) * (xin[1] / 4)
            + (sign(ang) * xin[1] / 8 if abs(ang) > 0.6 else 0)
        )
        norm_sol = lambda xin, y2, ang: pseudonorm(ang, xin) - (
            sign(ang) * (xin[1] / 4) + (sign(ang) * xin[1] / 8 if abs(ang) > 0.6 else 0)
        )
        z2 = x[0] * theta + x[1] * cos_est(theta)
        z1 = norm_sol(x, z2, theta)
        return np.array([z1, z2])


