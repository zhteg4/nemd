# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module calculates thermal conductivity via non-equilibrium thermodynamics.
"""
from scipy import constants


class ThermalConductivity(object):

    def __init__(self, temp_gradient, energy_flow, cross_sectional_area):
        self.temp_gradient = temp_gradient
        self.energy_flow = energy_flow
        self.cross_sectional_area = cross_sectional_area
        self.thermal_conductivity = None

    def run(self):
        if not all(
            [self.temp_gradient, self.energy_flow, self.cross_sectional_area]):
            return

        temp_gradient = abs(
            self.temp_gradient)  # Temperature (K) / Coordinate (Angstrom)
        temp_gradient_iu = temp_gradient / constants.angstrom
        # Energy (Kcal/mole) / Time (ns)
        heat_flow_ui = self.energy_flow * constants.calorie / constants.N_A * 1000 / constants.nano
        cross_section = self.cross_sectional_area  # Angstrom^2
        cross_section_ui = cross_section * (constants.angstrom**2)
        # Fourier's law qx = -k * dT / dx
        self.thermal_conductivity = heat_flow_ui / cross_section_ui / abs(
            temp_gradient_iu)
