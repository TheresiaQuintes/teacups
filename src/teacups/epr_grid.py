#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022.

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import numpy as np
from scipy.spatial import SphericalVoronoi, geometric_slerp
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, grid: str = "SOPHE", point_group="Ci", knots: int = 15):
        self._symmetry = point_group

        self._points = int(knots)

        if grid.lower() == "sophe":
            self._get_SOPHE_grid()
        else:
            raise ValueError("{} is not a valid grid!".format(grid))

        self._cartesian = False
        self._sv = None

    def get_grid(self, point_group: str = "Ci", cartesian: bool = False) -> np.array:
        """
        Get the grid for a given point group.

        Parameters
        ----------
        point_group : str, optional
            Symmetry. The default is "Ci". 'C1' returns the full sphere.
        cartesian : bool, optional
            If True, the coordinates will be returned as cartesians. The
            default is False.

        Returns
        -------
        np.array, (N, 3)
            Grid coordinates.

        """
        if self._symmetry != point_group:
            self._symmetry = point_group
            self._get_SOPHE_grid()
            self._sv = None
            self._cartesian = False

        if cartesian and not self._cartesian:
            self._grid = spherical_to_cartesian(self._grid[:, 1], self._grid[:, 2])
            self._border_points = spherical_to_cartesian(
                self._border_points[:, 1], self._border_points[:, 2]
            )
            self._cartesian = True

        return self._grid

    def get_areas(self) -> np.array:
        """
        Get the corresponding areas

        Returns
        -------
        np.array, (N,)
            Area of each Voronoi cell.

        """
        if self._symmetry in ["O3", "Dooh"]:
            return self._weight_factors
        else:
            areas = self._get_Voronoi_area() * self._weight_factors
        return areas

    def show(self, voronoi=False):
        if voronoi and self._sv is not None:
            self._sv.sort_vertices_of_regions()
            if not self._cartesian:
                grid = spherical_to_cartesian(self._grid[:, 1], self._grid[:, 2])
            else:
                grid = self._grid

            t_vals = np.linspace(0, 1, 2000)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # plot the unit sphere for reference (optional)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color="y", alpha=0.1)

            # Plot all points with color indication for their weight
            g_1 = grid[self._weight_factors == 1]
            g_2 = grid[self._weight_factors == 0.5]
            g_3 = grid[self._weight_factors == 0.25]
            g_4 = grid[self._weight_factors == 1 / 6]
            g_5 = grid[self._weight_factors == 1 / 3]

            ax.scatter(g_1[:, 0], g_1[:, 1], g_1[:, 2], color="b")
            ax.scatter(g_2[:, 0], g_2[:, 1], g_2[:, 2], color="k")
            ax.scatter(g_3[:, 0], g_3[:, 1], g_3[:, 2], color="y")
            ax.scatter(g_4[:, 0], g_4[:, 1], g_4[:, 2], color="r")
            ax.scatter(g_5[:, 0], g_5[:, 1], g_5[:, 2], color="m")

            # plot Voronoi vertices
            ax.scatter(
                self._sv.vertices[:, 0],
                self._sv.vertices[:, 1],
                self._sv.vertices[:, 2],
                c="g",
            )

            # indicate Voronoi regions (as Euclidean polygons)
            for region in self._sv.regions:
                n = len(region)
                for i in range(n):
                    start = self._sv.vertices[region][i]
                    end = self._sv.vertices[region][(i + 1) % n]
                    result = geometric_slerp(start, end, t_vals)
                    ax.plot(result[..., 0], result[..., 1], result[..., 2], c="k")

            ax.azim = 10
            ax.elev = 40
            _ = ax.set_xticks([])
            _ = ax.set_yticks([])
            _ = ax.set_zticks([])
            fig.set_size_inches(4, 4)

        else:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(self._grid[:, 0], self._grid[:, 1], self._grid[:, 2])

        plt.show()

    def _get_SOPHE_grid(self):
        r"""
        Get a SOPHE grid for one octant.

        .. math::

            \theta_{k, l}&=\frac{k}{M}\cdot\frac{\pi}{2}\;\;\;\;0\leq k\leq M\\
            \phi_{k, l} & = \frac{l}{k}\cdot\frac{\pi}{2}\;\;\;\; 0\leq l\leq k

        Parameters
        ----------
        M : float
            Size. Return N = (M+1)*(M+2)/2 knots.

        Returns
        -------
        points : np.array, (N, 3)
            Spherical coordinates of the grid. (Radius, Elevation, Azimuth).
        """
        phi_max, octants, border = self._get_grid_params()
        M = self._points

        # 03
        if octants == -1:
            theta = np.zeros(1)
            phi = np.zeros(1)
            r = np.ones(1)
            weights = np.ones(1)
            self._grid = np.array([r, theta, phi]).T
            self._weight_factors = weights
            self._border_points = None
            return

        # Dooh
        elif octants == 0:
            dtheta = np.pi / (2 * (M - 1))
            theta = np.linspace(0, M - 1, M) * dtheta
            phi = np.zeros(M)
            r = np.ones(M)
            weights = 4 * np.pi * np.sin(dtheta / 2) * np.sin(theta)
            weights[0] = 2 * np.pi * (1 - np.cos(dtheta / 2))
            weights[-1] = 2 * np.pi * np.sin(dtheta / 2)
            self._grid = np.array([r, theta, phi]).T
            self._weight_factors = weights
            self._border_points = None
            return

        else:
            # Pole
            theta = [0]
            phi = [0]
            weight_fac = [phi_max / (2 * np.pi)]

        # Coordinates without Voronoi border points
        if octants != 4 and octants != 8:
            for k in range(1, M + 1):
                for l in range(0, octants * k + border):
                    t = (k / M) * (np.pi / 2)
                    p = (l / (octants * k)) * phi_max

                    if border:
                        if l == 0 or l == octants * k:
                            weight = 0.5
                        else:
                            weight = 1.0
                    else:
                        weight = 1.0

                    if k == M:
                        weight *= 0.5

                    theta.append(t)
                    phi.append(p)
                    weight_fac.append(weight)

        else:
            for k in range(1, M + 1):
                for l in range(0, 4 * k):
                    t = (k / M) * (np.pi / 2)
                    p = (l / (4 * k)) * phi_max

                    if k == M and octants == 4:
                        weight = 0.5
                    else:
                        weight = 1.0

                    theta.append(t)
                    phi.append(p)
                    weight_fac.append(weight)

        theta = np.array(theta)
        phi = np.array(phi)
        self._weight_factors = np.array(weight_fac)

        if octants == 8:
            theta_2 = np.pi - theta[theta != np.pi / 2]
            phi_2 = phi[theta != np.pi / 2]
            theta = np.hstack((theta, theta_2))
            phi = np.hstack((phi, phi_2))
            self._weight_factors = np.ones(theta.size)

        r = np.ones(theta.size)
        self._grid = np.array([r, theta, phi]).T

        # Add border points for Voronoi area calculation
        theta = []
        phi = []
        if octants != 4 and octants != 8:
            for k in range(1, M + 2):
                if k == 1:
                    n_points = octants
                    max_points = int(((2 * np.pi) / phi_max) * n_points)
                    lb = -(max_points - n_points - 1 - border)
                else:
                    lb = -1
                for l in range(lb, octants * k + 1 + border):
                    if k != M + 1 and l > -1 and l < octants * k + border:
                        continue
                    elif k != M + 1:
                        t = (k / M) * (np.pi / 2)
                        p = (l / (octants * k)) * phi_max
                    else:
                        if l > (k - 2) * octants + border:
                            continue
                        t = (k / M) * (np.pi / 2)
                        p = (l / (octants * (M - 1))) * phi_max

                    theta.append(t)
                    phi.append(p)

        elif octants == 4:
            k = M - 1
            for l in range(0, 4 * k):
                t = np.pi - (k / M) * (np.pi / 2)
                p = (l / (octants * k)) * phi_max

                theta.append(t)
                phi.append(p)

        theta = np.array(theta)
        phi = np.array(phi)
        r = np.ones(theta.size)
        self._border_points = np.array([r, theta, phi]).T

    def _get_EasySpin_grid(self, M: int) -> np.array:
        r"""
        Get an EasySpin grid for one octant.

        .. math::

            \theta_A &= \frac{k}{M}\cdot\frac{\pi}{2} \\
            \phi_A &= \frac{l}{k}\cdot\frac{\pi}{2} \\
            \theta_B &= \frac{M-l}{M}\cdot\frac{\pi}{2} \\
            \phi_B &= \frac{k-l}{M-l}\cdot\frac{\pi}{2} \\
            \theta_C &= \frac{M-k+l}{M}\cdot\frac{\pi}{2} \\
            \phi_C &= \frac{M-k}{M-k+l}\cdot\frac{\pi}{2}

        .. math::

            \begin{pmatrix}x_i\\y_i\\z_i\end{pmatrix} &= \begin{pmatrix}
            \sin(\theta_i)\cos(\phi_i)\\\sin(\theta_i)\sin(\phi_i)\\
                \cos(\theta_i)\end{pmatrix}\\
            \begin{pmatrix}x\\y\\z\end{pmatrix} &= \frac{1}{3}\left(
            \begin{pmatrix}x_A\\y_A\\z_A\end{pmatrix}+
            \begin{pmatrix}y_B\\z_B\\x_B\end{pmatrix}+
            \begin{pmatrix}z_C\\x_C\\y_C\end{pmatrix}\right)


        Parameters
        ----------
        M : int
            Size. Return N = (M+1)*(M+2)/2 knots.

        Returns
        -------
        points : np.array, (N, 3)
            Spherical coordinates of the grid. (Radius, Elevation, Azimuth).

        """
        k, l = np.tril_indices(M + 1)
        theta_a = (k / M) * (np.pi / 2)
        phi_a = (l / k) * (np.pi / 2)
        phi_a = np.nan_to_num(phi_a, nan=np.pi / 2)

        theta_b = ((M - l) / M) * (np.pi / 2)
        phi_b = ((k - l) / (M - l)) * (np.pi / 2)
        phi_b = np.nan_to_num(phi_b, nan=np.pi / 2)

        theta_c = ((M - k + l) / M) * (np.pi / 2)
        phi_c = ((M - k) / (M - k + l)) * (np.pi / 2)
        phi_c = np.nan_to_num(phi_c, nan=np.pi / 2)

        x_a = np.sin(theta_a) * np.cos(phi_a)
        y_a = np.sin(theta_a) * np.sin(phi_a)
        z_a = np.cos(theta_a)

        x_b = np.sin(theta_b) * np.cos(phi_b)
        y_b = np.sin(theta_b) * np.sin(phi_b)
        z_b = np.cos(theta_b)

        x_c = np.sin(theta_c) * np.cos(phi_c)
        y_c = np.sin(theta_c) * np.sin(phi_c)
        z_c = np.cos(theta_c)

        x = (x_a + y_b + z_c) / 3
        y = (y_a + z_b + x_c) / 3
        z = (z_a + x_b + y_c) / 3

        points = cartesian_to_spherical(x, y, z)
        return points

    def _get_Voronoi_area(self) -> np.array:
        """
        Get the area of the Voronoi cell for each coordinate on the sphere.

        Parameters
        ----------
        coordinates : np.array, (N, 3)
            Coordinates of the sphere.

        Returns
        -------
        areas : np.array, (N, )
            Area for each coordinate.

        """
        cartesian = self._cartesian
        coordinates = np.vstack((self._grid, self._border_points))

        # Transform to spherical and back to obtain a consistent radius
        if cartesian:
            coordinates = cartesian_to_spherical(
                coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
            )
            coordinates = spherical_to_cartesian(coordinates[:, 1], coordinates[:, 2])

        # Transform to cartesian
        coordinates = spherical_to_cartesian(coordinates[:, 1], coordinates[:, 2])

        self._sv = SphericalVoronoi(coordinates, radius=1)
        areas = self._sv.calculate_areas()

        return areas[: self._grid[:, 0].size]

    def _get_grid_params(self) -> [float, int, bool]:
        point_group = [
            "C1",
            "Ci",
            "C2h",
            "S6",
            "C4h",
            "C6h",
            "D2h",
            "Th",
            "D3d",
            "D4h",
            "Oh",
            "D6h",
            "Dooh",
            "O3",
        ]
        pg_idx_dic = dict(zip(point_group, np.linspace(0, 13, 14, dtype=np.int8)))

        # phi in 1/4 *pi
        phi = [8, 8, 4, 8 / 3, 2, 4 / 3, 2, 2, 4 / 3, 1, 1, 2 / 3, 0, 0]
        bounds = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        octants = [8, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1]

        idx = pg_idx_dic[self._symmetry]

        return phi[idx] * np.pi / 4, octants[idx], bool(bounds[idx])

    def get_triangle_idx(self) -> np.array:
        # TODO: Ordentlich machen und optimieren!
        """
        Get the indices for all possible triangle of the current grid.

        Returns
        -------
        triangles : np.array, (N, 4)
            Indices for theta and phi to obtain a triangle on the grid. Also
            contains the area of the Delaunay triangle.
            (idx1, idx2, idx3, area)

        """
        phi_max, octants, border = self._get_grid_params()
        M = self._points

        # u, v are parameterisation variables
        r = self._grid[:, 0]
        u = self._grid[:, 1]
        v = self._grid[:, 2]

        xyz = spherical_to_cartesian(u, v)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        if octants == 1:
            uptris = []
            downtris = []
            for n in range(M):
                if n == 0:
                    uprow = np.array([0])
                    downrow = np.linspace(1, 1 + border, 1 + border, dtype=np.int32)
                    start_2 = 1
                    stop_2 = 1 + border
                    uptris.append([0, downrow[0], downrow[1 % (1 + border)]])
                    continue
                else:
                    start = 1 * start_2
                    stop = 1 * stop_2
                    start_2 = stop + 1
                    stop_2 = start_2 + (n + border)
                    uprow = np.linspace(start, stop, stop - start + 1, dtype=np.int32)
                    downrow = np.linspace(
                        start_2, stop_2, stop_2 - start_2 + 1, dtype=np.int32
                    )

                for i in range(len(uprow) - border):
                    idx_top_left = uprow[i]
                    idx_top_right = uprow[(i + 1) % len(uprow)]
                    idx_bottom = downrow[(i + 1) % len(downrow)]
                    downtris.append([idx_top_left, idx_top_right, idx_bottom])

                for i in range(len(downrow) - border):
                    idx_top = uprow[i % len(uprow)]
                    idx_down_left = downrow[i]
                    idx_down_right = downrow[(i + 1) % len(downrow)]
                    uptris.append([idx_top, idx_down_left, idx_down_right])

        elif octants == 2:
            uptris = []
            downtris = []
            for n in range(M):
                if n == 0:
                    uprow = np.array([0])
                    downrow = np.linspace(1, 2, 2, dtype=np.int32)
                    start_2 = 1
                    stop_2 = 2
                    for i in range(octants):
                        uptris.append([0, downrow[i], downrow[(i + 1) % octants]])
                    continue
                else:
                    start = 1 * start_2
                    stop = 1 * stop_2
                    start_2 = stop + 1
                    stop_2 = start_2 + (n + 1) * octants - 1
                    uprow = np.linspace(start, stop, stop - start + 1, dtype=np.int32)
                    downrow = np.linspace(
                        start_2, stop_2, stop_2 - start_2 + 1, dtype=np.int32
                    )

                for k in range(2):
                    for i in range(1, len(uprow) // octants + 1):
                        offset_up = k * n
                        offset_low = k * (n + 1)
                        idx_top_left = uprow[(i - 1 + offset_up) % len(uprow)]
                        idx_top_right = uprow[(i + offset_up) % len(uprow)]
                        idx_bottom = downrow[(i + offset_low) % len(downrow)]
                        downtris.append([idx_top_left, idx_top_right, idx_bottom])

                    for i in range(len(downrow) // octants):
                        offset_up = k * n
                        offset_low = k * (n + 1)
                        idx_top = uprow[(i + offset_up) % len(uprow)]
                        idx_down_left = downrow[(i + offset_low) % len(downrow)]
                        idx_down_right = downrow[(i + 1 + offset_low) % len(downrow)]
                        uptris.append([idx_top, idx_down_left, idx_down_right])

        elif octants == 4:
            uptris = []
            downtris = []
            for n in range(M):
                if n == 0:
                    uprow = np.array([0])
                    downrow = np.linspace(1, 4, 4, dtype=np.int32)
                    start_2 = 1
                    stop_2 = 4
                    for i in range(octants):
                        uptris.append([0, downrow[i], downrow[(i + 1) % octants]])
                    continue
                else:
                    start = 1 * start_2
                    stop = 1 * stop_2
                    start_2 = stop + 1
                    stop_2 = start_2 + (n + 1) * octants - 1
                    uprow = np.linspace(start, stop, stop - start + 1, dtype=np.int32)
                    downrow = np.linspace(
                        start_2, stop_2, stop_2 - start_2 + 1, dtype=np.int32
                    )

                for k in range(4):
                    for i in range(1, len(uprow) // octants + 1):
                        offset_up = k * n
                        offset_low = k * (n + 1)
                        idx_top_left = uprow[(i - 1 + offset_up) % len(uprow)]
                        idx_top_right = uprow[(i + offset_up) % len(uprow)]
                        idx_bottom = downrow[(i + offset_low) % len(downrow)]
                        downtris.append([idx_top_left, idx_top_right, idx_bottom])

                    for i in range(len(downrow) // octants):
                        offset_up = k * n
                        offset_low = k * (n + 1)
                        idx_top = uprow[(i + offset_up) % len(uprow)]
                        idx_down_left = downrow[(i + offset_low) % len(downrow)]
                        idx_down_right = downrow[(i + 1 + offset_low) % len(downrow)]
                        uptris.append([idx_top, idx_down_left, idx_down_right])

        elif octants == 8:
            uptris = []
            downtris = []
            for hemi in range(2):
                for n in range(M):
                    if n == 0:
                        if hemi == 0:
                            hemi_offset = 0
                        else:
                            hemi_offset = stop_2 + 1
                            replace_line = downrow.copy()
                        uprow = np.array([0 + hemi_offset])
                        downrow = np.linspace(
                            1 + hemi_offset, 4 + hemi_offset, 4, dtype=np.int32
                        )
                        start_2 = 1 + hemi_offset
                        stop_2 = 4 + hemi_offset
                        for i in range(4):
                            uptris.append(
                                [
                                    0 + hemi_offset,
                                    downrow[i],
                                    downrow[(i + 1) % 4],
                                ]
                            )
                        continue
                    else:
                        start = 1 * start_2
                        stop = 1 * stop_2
                        start_2 = stop + 1
                        stop_2 = start_2 + (n + 1) * 4 - 1
                        uprow = np.linspace(
                            start, stop, stop - start + 1, dtype=np.int32
                        )
                        if n == M - 1 and hemi == 1:
                            downrow = replace_line
                        else:
                            downrow = np.linspace(
                                start_2,
                                stop_2,
                                stop_2 - start_2 + 1,
                                dtype=np.int32,
                            )

                    for k in range(4):
                        for i in range(1, len(uprow) // 4 + 1):
                            offset_up = k * n
                            offset_low = k * (n + 1)
                            idx_top_left = uprow[(i - 1 + offset_up) % len(uprow)]
                            idx_top_right = uprow[(i + offset_up) % len(uprow)]
                            idx_bottom = downrow[(i + offset_low) % len(downrow)]
                            downtris.append([idx_top_left, idx_top_right, idx_bottom])

                        for i in range(len(downrow) // 4):
                            offset_up = k * n
                            offset_low = k * (n + 1)
                            idx_top = uprow[(i + offset_up) % len(uprow)]
                            idx_down_left = downrow[(i + offset_low) % len(downrow)]
                            idx_down_right = downrow[
                                (i + 1 + offset_low) % len(downrow)
                            ]
                            uptris.append([idx_top, idx_down_left, idx_down_right])

        uptris = np.array(uptris)
        downtris = np.array(downtris)
        indices = np.vstack([uptris, downtris])

        def spherical_area(a, b, c):
            t = abs(np.inner(a, np.cross(b, c)))
            t /= 1 + np.inner(a, b) + np.inner(b, c) + np.inner(a, c)
            return 2 * np.arctan(t)

        # Correct the triangles for area calculation if border is False
        if octants == 8:
            border = True
        if not border:
            idx_dt = indices[:, 0] >= indices[:, 1]
            idx_ut = indices[:, 1] >= indices[:, 2]

        areas = []
        points = []
        for i in range(len(indices)):
            if not border and idx_dt[i]:
                coords = xyz[indices[i, 1]]
                coords = cartesian_to_spherical(*coords)
                coords[2] = phi_max
                coords = spherical_to_cartesian(coords[1], coords[2])

                p_1 = np.array(xyz[indices[i, 0]])
                p_2 = coords
                p_3 = np.array(xyz[indices[i, 2]])

            elif not border and idx_ut[i]:
                coords_1 = np.array(xyz[indices[i, 0]])
                coords_1 = cartesian_to_spherical(*coords_1)
                coords_1[2] = phi_max
                coords_1 = spherical_to_cartesian(coords_1[1], coords_1[2])

                coords_2 = xyz[indices[i, 2]]
                coords_2 = cartesian_to_spherical(*coords_2)
                coords_2[2] = phi_max
                coords_2 = spherical_to_cartesian(coords_2[1], coords_2[2])
                p_1 = coords_1
                p_2 = np.array(xyz[indices[i, 1]])
                p_3 = coords_2
            else:
                p_1 = np.array(xyz[indices[i, 0]])
                p_2 = np.array(xyz[indices[i, 1]])
                p_3 = np.array(xyz[indices[i, 2]])

            areas.append(spherical_area(p_1, p_2, p_3))

        indices = np.hstack([indices, np.array([areas]).T])

        return indices


def spherical_to_cartesian(
    theta: np.array, phi: np.array, r: np.array = 1.0
) -> np.array:
    r"""
    Transform spherical coordinates to cartesian.

    .. math::

        x &= r\cdot\sin(\theta)\cdot\cos(\phi) \\
        y &= r\cdot\sin(\theta)\cdot\sin(\phi) \\
        z &= r\cdot\cos(\theta)

    Parameters
    ----------
    theta : np.array, float
        N theta angles. Elevation.
    phi : np.array, float
        N phi angles. Azimuth.
    r : np.array, float, optional
        Radius of the sphere.

    Returns
    -------
    np.array, (N, 3)
        Transformed coordinates.

    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z]).T


def cartesian_to_spherical(x: np.array, y: np.array, z: np.array) -> np.array:
    r"""
    Transform cartesian coordinates to spherical.

    .. math::

        r &= \sqrt{x^2 +y^2 + z^2} \\
        \theta &= \arccos(\frac{z}{r})\\
        \phi &= \arctan2(y, x)

    Parameters
    ----------
    x : np.array
        x values.
    y : np.array
        y values.
    z : np.array
        z values.

    Returns
    -------
    np.array, (N, 3)
        Transformed coordinates (r, \theta, \phi) -> (radius, elevation,
        azimuth).

    Note
    ----
    This convention uses a right-handed coordinate system.

    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi]).T
