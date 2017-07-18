#!/usr/bin/env python
"""
An emulation engine for KaFKA. This emulation engine is designed to be useful 
for the atmospheric correction part.
"""

# KaFKA A fast Kalman filter implementation for raster based datasets.
# Copyright (c) 2017 J Gomez-Dans. All rights reserved.
#
# This file is part of KaFKA.
#
# KaFKA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KaFKA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KaFKA.  If not, see <http://www.gnu.org/licenses/>.


import os
import glob
import sys
import cPickle

import numpy as np

import gp_emulator # unnecessary?


__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (13.07.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"

class AtmosphericEmulationEngine(object):
    """An emulation engine for single band atmospheric RT models.
    For reference, the ordering of the emulator parameters is
    1. cos(sza)
    2. cos(vza)
    3. saa (in degrees)
    4. vaa (in degrees)
    5. AOT@550
    6. Water Vapour (in 6s units, cm)
    7. Ozone concentration (in 6S units, cm-atm)
    8. Altitude (in km)
    """
    def __init__ ( self, sensor, emulator_folder):
        self.sensor = sensor
        self._locate_emulators(sensor, emulator_folder)

    def _locate_emulators(self, sensor, emulator_folder):
        self.emulators = []
        self.emulator_names = []
        files = glob.glob(os.path.join(emulator_folder, 
                "*%s*.pkl" % sensor))
        files.sort()
        for fich in files:
            emulator_file = os.path.basename(fich)
            # Create an emulator label (e.g. band name)
            self.emulator_names = emulator_file
            self.emulators.append ( cPickle.load(open(fich, 'r')))
            log.info("Found file %s, storing as %s" %
                        fich, emulator_file)
        self.n_bands = len(self.emulators)

    def emulator_kernel_atmosphere(self, kernel_weights, atmosphere, 
                sza, vza, saa, vaa, elevation, 
                gradient_kernels=True, bands=None):
        """This method puts together a 2D array with the parameters
        for the emulator. This method takes kernel weights for
        different bands (iso, vol, geo for self.n_bands),
        atmospheric parameters (AOT, TCWV and O3), as well as some
        "control" variables (view/illumination angles and elevation).
        The method returns the forward modelled TOA reflectances and
        the associated Jacobian. If the option `gradient_kernels` is 
        set to `True`, the Jacobian will also be calculated for the kernels 
        (e.g. in the case of minimising a combined cost function of
        atmosphere and land surface).
        
        We expect `kernel_weights` to be a `3 x self.n_bands x n_pixels`
        array, `atmosphere` to be a `3 x n_pixels` array, and vza, sza,
        vaa, saa and elevation to be `n_pixels` arrays, or if they are
        assumed constaint, they can be scalars.
        
        The `bands` option is there to select individual bands, and it
        should either be a scalar (with the band position in the 
        emulator array), or a list (again with band positions). In this
        case, the kernels can be passed only for the band(s) that are
        requested, but in the same order as the bands. E.g. if 
        `bands=[3,4,5]`, then `kernel_weights` bands should also be
        ordered as band positions 3, 4 and 5 along the second axis.
        """
        
        # the controls can be scalars or arrays
        # We convert them to arrays if needed
        sza = np.asarray(sza).reshape(1, -1)[0,:]
        vza = np.asarray(vza).reshape(1, -1)[0,:]
        saa = np.asarray(saa).reshape(1, -1)[0,:]
        vaa = np.asarray(vaa).reshape(1, -1)[0,:]
        elevation = np.asarray(elevation).reshape(1, -1)[0,:]
        # the mother of all arrays will be 3*nbands+3+4
        n_pix1 = kernel_weights.shape[2]
        n_pix2 = atmosphere.shape[1]
        assert n_pix1 == n_pix2  # In reality could check angles and stuff
        n_pix = n_pix1 
        x = np.zeros((8 + 3, n_pix)) # 11 parameters innit?
        x[3:, :] = np.c_[np.cos(sza)*np.ones(n_pix), 
                         np.cos(vza)*np.ones(n_pix), 
                         saa*np.ones(n_pix), vaa*np.ones(n_pix), 
                         atmosphere[0,:], atmosphere[1,:], atmosphere[2,:], 
                         elevation*np.ones(n_pix)].T
        
        H0 = []
        dH = []
        if bands is None: # Do all bands
            for band in xrange(self.n_bands):
                emu = self.emulators[band]
                x[0, :] = kernel_weights[0, band, :] # Iso
                x[1, :] = kernel_weights[1, band, :] # Vol
                x[2, :] = kernel_weights[2, band, :] # Geo
                H0_, dH_ = emu.predict(x, do_unc=False)
                if not gradient_kernels:
                    dH_ = dH_[3:, :] # Ignore the kernels in the gradient 
                H0.append(H0_)
                dH.append(dH_)
                
        else:
            # This is needed in case we get a single band
            the_bands = (bands,) if not isinstance(bands, 
                                                   (tuple, list)) else bands
            if max(the_bands) > (self.n_bands-1):
                raise ValueError("There are only " + 
                    "%d bands, and you asked for %d position" 
                    % (self.n_bands, max(the_bands)))
            sel_bands = len(the_bands)
            if kernel_weights.shape[1] == sel_bands:
                # We only got passed a subset of the bands
                is_subset = True
            else:
                is_subset = False
            for j, band in enumerate(the_bands):
                emu = self.emulators[band]
                if is_subset:
                    x[0, :] = kernel_weights[0, j, :] # Iso
                    x[1, :] = kernel_weights[1, j, :] # Vol
                    x[2, :] = kernel_weights[2, j, :] # Geo                    
                else:
                    x[0, :] = kernel_weights[0, band, :] # Iso
                    x[1, :] = kernel_weights[1, band, :] # Vol
                    x[2, :] = kernel_weights[2, band, :] # Geo
                H0_, dH_ = emu.predict(x, do_unc=False)
                if not gradient_kernels:
                    dH_ = dH_[3:, :] # Ignore the kernels in the gradient 
                H0.append(H0_)
                dH.append(dH_)
        return H0, dH
            
            
    def emulator_reflectance_atmosphere(self, reflectance, atmosphere, 
                sza, vza, saa, vaa, elevation, 
                gradient_refl=True, bands=None):
        """This method puts together a 2D array with the parameters
        for the emulator. This method takes SDR for
        different bands, atmospheric parameters (AOT, TCWV and O3), 
        as well as some "control" variables (view/illumination angles 
        and elevation).
        
        The method returns the forward modelled TOA reflectances and
        the associated Jacobian. If the option `gradient_refl` is 
        set to `True`, the Jacobian will also be calculated for the kernels 
        (e.g. in the case of minimising a combined cost function of
        atmosphere and land surface).
        
        We expect `reflectance` to be a `self.n_bands x n_pixels`
        array, `atmosphere` to be a `3 x n_pixels` array, and vza, sza,
        vaa, saa and elevation to be `n_pixels` arrays, or if they are
        assumed constaint, they can be scalars.
        
        The `bands` option is there to select individual bands, and it
        should either be a scalar (with the band position in the 
        emulator array), or a list (again with band positions). In this
        case, the reflectance be passed only for the band(s) that are
        requested, but in the same order as the bands. E.g. if 
        `bands=[3,4,5]`, then `reflectance` bands should also be
        ordered as band positions 3, 4 and 5 along the second axis.
        """
        # the controls can be scalars or arrays
        # We convert them to arrays if needed
        sza = np.asarray(sza).reshape(1, -1)[0,:]
        vza = np.asarray(vza).reshape(1, -1)[0,:]
        saa = np.asarray(saa).reshape(1, -1)[0,:]
        vaa = np.asarray(vaa).reshape(1, -1)[0,:]

        # the mother of all arrays will be 3*nbands+3+4
        n_pix1 = reflectance.shape[1]
        n_pix2 = atmosphere.shape[1]
        assert n_pix1 == n_pix2  # In reality could check angles and stuff
        n_pix = n_pix1 
        x = np.zeros((9, n_pix)) # 10 parameters innit?
        x[1:, :] = np.c_[np.cos(sza)*np.ones(n_pix), 
                         np.cos(vza)*np.ones(n_pix), 
                         saa*np.ones(n_pix), vaa*np.ones(n_pix), 
                         atmosphere[0,:], atmosphere[1,:], atmosphere[2,:], 
                         elevation*np.ones(n_pix)].T
        H0 = []
        dH = []
        if bands is None: # Do all bands
            for band in xrange(self.n_bands):
                emu = self.emulators[band]
                x[0, ] = reflectance[band, :]
                H0_, dH_ = emu.predict(x, do_unc=False)
                if not gradient_refl:
                    dH_ = dH_[1:, :] # Ignore the SDR in the gradient 
                H0.append(H0_)
                dH.append(dH_)
                
        else:
            # This is needed in case we get a single band
            the_bands = (bands,) if not isinstance(bands, 
                                                   (tuple, list)) else bands
            if max(the_bands) > (self.n_bands-1):
                raise ValueError("There are only " + 
                    "%d bands, and you asked for %d position" 
                    % (self.n_bands, max(the_bands)))

            sel_bands = len(the_bands)
            if reflectance.shape[0] == sel_bands:
                # We only got passed a subset of the bands
                is_subset = True
            else:
                is_subset = False
            for j, band in enumerate(the_bands):
                emu = self.emulators[band]
                if is_subset:
                    x[0, :] = reflectance[j, :] 
                else:
                    x[0, :] = reflectance[band, :]
                H0_, dH_ = emu.predict(x, do_unc=False)
                if not gradient_refl:
                    dH_ = dH_[1:, :] # Ignore the SDR in the gradient 
                H0.append(H0_)
                dH.append(dH_)
        return H0, dH            
            
            
            
            
        
        
        
