import numpy as np
import teacups.grid as grid


class TestSphereFibonacciGridPoints:
    def setup(self):
        self.npoints = 30
        self.hemisphere = grid.sphere_fibonacci_grid_points(self.npoints)
        self.sphere = grid.sphere_fibonacci_grid_points(self.npoints,
                                                        hemisphere=False)

    def test_shape_hemisphere(self):
        assert self.hemisphere.shape == (30, 3)

    def test_shape_sphere(self):
        assert self.sphere.shape == (30, 3)

    def test_hemisphere(self):
        h = grid.sphere_fibonacci_grid_points(int(self.npoints/2))
        f = self.sphere[0:int(self.npoints/2)]
        assert np.array_equal(h, f)


class TestCartesian2Spherical:
    def setup(self):
        self.xyz = np.array([[2, 3, 4]])
        self.rtp = grid.cartesian2sphereical(self.xyz)

    def test_transformation(self):
        r = np.sqrt(self.xyz[:, 0]**2 + self.xyz[:, 1]**2 + self.xyz[:, 2]**2)
        theta = np.arccos(self.xyz[:, 2]/r)
        phi = np.arctan(self.xyz[:, 1]/self.xyz[:, 0])
        rtp = np.array([r, theta, phi]).T
        assert np.array_equal(rtp, self.rtp)

    def test_on_x(self):
        function = grid.cartesian2sphereical(np.array([[1, 0, 0]]))
        target = np.array([[1,np.pi/2, 0]])
        assert np.array_equal(function, target)

    def test_on_y(self):
        function = grid.cartesian2sphereical(np.array([[0, 1, 0]]))
        target = np.array([[1, np.pi/2, np.pi/2]])
        assert np.array_equal(function, target)

    def test_on_z(self):
        function = grid.cartesian2sphereical(np.array([[0, 0, 1]]))
        target = np.array([[1, 0, 0]])
        assert np.array_equal(function, target)


class TestSpherical2Cartesian:
    def setup(self):
        self.xyz = np.array([[2., 3., 4.]])
        self.rtp = grid.cartesian2sphereical(self.xyz)

    def test_retransformation(self):
        xyz = grid.spherical2cartesian(self.rtp)
        assert np.allclose(xyz, self.xyz)


class TestFibonacciGrid:
    def setup(self):
        self.grid_points = 30
        self.theta, self.phi = grid.fibonacci_grid(self.grid_points)

    def test_shape(self):
        assert self.theta.shape == (self.grid_points,)

    def test_shape_phi(self):
        assert self.phi.shape == (self.grid_points, )

    def test_only_upper_hemisphere(self):
        assert np.array_equal(self.theta, abs(self.theta))


class TestSopheGrid:
    def setup(self):
        self.grid_size = 30
        self.theta, self.phi, self.weights = grid.sophe_grid(self.grid_size,
                                                             "C1")
    def test_shapes(self):
        assert self.theta.shape == self.phi.shape
        assert self.phi.shape == self.weights.shape
        assert self.weights.shape == (3602, )
