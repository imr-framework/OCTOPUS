<p align="center">
<img src="OCTOPUSLogo_Rect.png"/>
</p>

# OCTOPUS: Off-resonance CorrecTion OPen-soUrce Software
`OCTOPUS` is an open-source tool that provides off-resonance correction  methods for  Magnetic Resonance (MR) images. In particular, the implemented techniques are Conjugate Phase Reconstruction (CPR)[[1]](#references), frequency-segmented CPR [[2]](#references) and Multi-Frequency Interpolation (MFI) [[3]](#references).

Off-resonance is a type of MR image artifact. It originates as an accumulation of phase from off-resonant spins along the read-out direction due to field inhomogeneities, tissue susceptibilities and chemical shift among other possible sources [[4]](#references). Long read-out k-space trajectories are therefore more prone to this artifact and its consequences on the image. The image effects are tipycally blurring and/or geometrical distortion, and consequently, quality deterioration [[5]](#references).

`OCTOPUS` leverages existing techniques and outputs artifact-corrected or mitigated image reconstruction given the raw data from the scanner, k-space trajectory and field map. It is targeted to MR scientists, researchers, engineers and students who work with off-resonance-prone trajectories, such as spirals.

To learn more about the used methods and their implementation visit the [API docs][api-docs].

## Installation
1. Install Python (>=Python 3.6)
2. Create and activate a virtual environment (optional but recommended)
3. Copy and paste this command in your terminal
```pip install MR-OCTOPUS```

**Otherwise, [skip the installation!]** <mark>Run `OCTOPUS` in your browser instead.</mark>

## Quick start
The [Examples folder] contains scripts and data to run off-resonance correction on numerical simulations and phantom images for different k-space trajectories and field maps.

After the [installation] is completed, download the [example data]. Now you can run two types of demos. More information about these and other experiments can be found in the [Wiki page].

### 1. Numerical simulations

`numsim_cartesian.py` and `numsim_spiral.py` run a forward model on a 192x192 Shepp-Logan phantom image. They simulate the off-resonance effect of a cartesian and spiral k-space trajectory, respectively, given a simulated field map.

With `OCTOPUS.Fieldmap.fieldmap_gen` you can experiment the effect of the type of field map and its frequency range on the output corrupted image.

The corrupted image is then corrected using CPR, fs-CPR and MFI and the results are displayed.

### 2. In vitro experiment
If you want to use `OCTOPUS` to correct real data, you can use `ORC_main.py` as a template.
1. Fill the `settings.ini` file with the paths for your inputs and outputs. NOTE: the default settings are configured to run the script using the sample data provided.
2. Input your field of view (FOV), gradient raster time (dt), and echo time (TE).
```python
FOV =   # meters
dt =    # seconds
TE =    # seconds
```
3. Check that the dimensions of your inputs agree.
	`rawdata` dims = `ktraj` dims
5. Specify the number of frequency segments for the fs-CPR and MFI methods
```python
Lx =    # L=Lmin * Lx
```
6. Run the script.
The program will display an image panel with the original image and the corrected versions.

## Skip the installation! - `OCTOPUS` in your browser

There's no need to go through the installation process. Using this [template][colab-template] you can now run off-resonance correction in your browser!

As a demo, you can use the [example data] provided for the [in vitro experiment].

## Contributing and Community guidelines
`OCTOPUS` adheres to a code of conduct adapted from the [Contributor Covenant] code of conduct.
Contributing guidelines can be found [here][contrib-guidelines].

## References
1. Maeda, A., Sano, K. and Yokoyama, T. (1988), Reconstruction by weighted correlation for MRI with time-varying gradients. IEEE Transactions on Medical Imaging, 7(1): 26-31. doi: 10.1109/42.3926
2. Noll, D. C., Pauly, J. M., Meyer, C. H., Nishimura, D. G. and Macovskj, A. (1992), Deblurring for non‐2D fourier transform magnetic resonance imaging. Magn. Reson. Med., 25: 319-333. doi:10.1002/mrm.1910250210
3. Man, L., Pauly, J. M. and Macovski, A. (1997), Multifrequency interpolation for fast off‐resonance correction. Magn. Reson. Med., 37: 785-792. doi:10.1002/mrm.1910370523
4. Noll, D. C., Meyer, C. H., Pauly, J. M., Nishimura, D. G. and Macovski, A. (1991), A homogeneity correction method for magnetic resonance imaging with time-varying gradients. IEEE Transactions on Medical Imaging, 10(4): 629-637. doi: 10.1109/42.108599
5. Schomberg, H. (1999), Off-resonance correction of MR images. IEEE Transactions on Medical Imaging, 18( 6): 481-495. doi: 10.1109/42.781014

[api-docs]: https://mr-octopus.readthedocs.io/en/latest/
[Contributor Covenant]: http://contributor-covenant.org
[contrib-guidelines]: https://github.com/imr-framework/OCTOPUS/blob/master/CONTRIBUTING.md
[installation]: #installation
[in vitro experiment]: #2-in-vitro-experiment
[Examples folder]: https://github.com/imr-framework/OCTOPUS/tree/master/OCTOPUS/Examples
[example data]: https://github.com/imr-framework/OCTOPUS/blob/master/OCTOPUS/Examples/examples_zip.zip
[colab-template]: https://colab.research.google.com/drive/1hEIj5LaF19yOaWkSqi2uWXyy3u6UgKoP?usp=sharing
[skip the installation!]: #skip-the-installation---octopus-in-your-browser
[Wiki page]: https://github.com/imr-framework/OCTOPUS/wiki
