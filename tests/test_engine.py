#!/usr/bin/env python

import numpy as np
import pytest

from emulation_engine import AtmosphericEmulationEngine


@pytest.fixture
def set_up_atmospheric_class():
    
    class MockEmulator(object):
        def predict(self, x, do_unc=False):
            return np.ones(x.shape[1]), np.ones_like(x)*0.5

            
    class MockAtmosphericEmulationEngine(AtmosphericEmulationEngine):
        def __init__ (self, sensor, emulator_folder):
            self.sensor = sensor
            self.emulator_names = ["band1", "band2", "band3"]
            self.emulators = [ MockEmulator(), MockEmulator(), MockEmulator()]
            self.n_bands = 3
    mock_class = MockAtmosphericEmulationEngine( "testme", "/tmp/")
    return mock_class

def test_kernel_emulators(set_up_atmospheric_class):
    emu = set_up_atmospheric_class
    # First test returning gradient with kernels
    kernel_weights = np.ones((3, 3, 100))*0.1
    atmosphere = np.ones((3,100))*0.2
    sza = 37.5
    vza = 0.0
    saa = 0.
    vaa = 0.
    elevation = 0.5
    gradient_kernel = True
    bands = None
    H, dH = emu.emulator_kernel_atmosphere(kernel_weights, atmosphere, 
                sza, vza, saa, vaa, elevation, 
                gradient_kernels=gradient_kernel, bands=bands)
    assert len(H) == 3 and np.allclose(H[0], np.ones((100)))  
    assert len(H) == 3 and np.allclose(dH[0], 0.5*np.ones((11,100)))

    # No kernels in gradient
    gradient_kernel = False
    H, dH = emu.emulator_kernel_atmosphere(kernel_weights, atmosphere, 
                sza, vza, saa, vaa, elevation, 
                gradient_kernels=gradient_kernel, bands=bands)
    assert len(H) == 3 and np.allclose(H[0], np.ones((100)))  
    assert len(H) == 3 and np.allclose(dH[0], 0.5*np.ones((8,100)))

    # No kernels in gradient, only band 2 (or 1 in python)
    gradient_kernel = False
    bands = 1
    H, dH = emu.emulator_kernel_atmosphere(kernel_weights, atmosphere, 
                sza, vza, saa, vaa, elevation, 
                gradient_kernels=gradient_kernel, bands=bands)
    assert len(H) == 1 and np.allclose(H[0], np.ones((100)))  
    assert len(H) == 1 and np.allclose(dH[0], 0.5*np.ones((8,100)))

    # No kernels in gradient, bands 0+1 do gradient
    gradient_kernel = True
    bands = [0,1]
    H, dH = emu.emulator_kernel_atmosphere(kernel_weights, atmosphere, 
                sza, vza, saa, vaa, elevation, 
                gradient_kernels=gradient_kernel, bands=bands)
    assert len(H) == 2 and np.allclose(H[0], np.ones((100)))  
    assert len(H) == 2 and np.allclose(dH[0], 0.5*np.ones((11,100)))

    
    with pytest.raises(ValueError):
        # No kernels in gradient, only band 5 (or 4 in python)
        # ... which doesn't exist
        gradient_kernel = False
        bands = 4
        H, dH = emu.emulator_kernel_atmosphere(kernel_weights, atmosphere, 
                sza, vza, saa, vaa, elevation, 
                gradient_kernels=gradient_kernel, bands=bands)

    # Test having an array of vza's
    kernel_weights = np.ones((3, 3, 100))*0.1
    atmosphere = np.ones((3,100))*0.2
    sza = 37.5*np.ones(100)
    vza = 0.0
    saa = 0.
    vaa = 0.
    elevation = 0.5
    gradient_kernel = True
    bands = None

    H, dH = emu.emulator_kernel_atmosphere(kernel_weights, atmosphere, 
                sza, vza, saa, vaa, elevation, 
                gradient_kernels=gradient_kernel, bands=bands)
    assert len(H) == 3 and np.allclose(H[0], np.ones((100)))  
    assert len(H) == 3 and np.allclose(dH[0], 0.5*np.ones((11,100)))
