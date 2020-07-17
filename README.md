# OCTOPUS: Off-resonance CorrecTion OPen-soUrce Software
OCTOPUS is an open-source tool that provides off-resonance correction  methods for  Magnetic Resonance (MR) images. In particular, the implemented techniques are Conjugate Phase Reconstruction (CPR)[[1]](#references), frequency-segmented CPR [[2]](#references) and Multi-Frequency Interpolation (MFI) [[3]](#references).

Off-resonance is a type of MR image artifact. It originates as an accumulation of phase from off-resonant spins along the read-out direction due to field inhomogeneities, tissue susceptibilities and chemical shift among other possible sources [[4]](#references). Long read-out k-space trajectories are therefore more prone to this artifact and its consequences on the image. The image effects are tipycally blurring and/or geometrical distortion, and consequently, quality deterioration [[5]](#references).

OCTOPUS leverages existing techniques and outputs artifact-corrected or mitigated image reconstruction given the raw data from the scanner, k-space trajectory and field map. It is targeted to MR scientists, researchers, engineers and students who work with off-resonance-prone trajectories, such as spirals.

To learn more about the used methods and their implementation visit the API docs.

## Contributing and Community guidelines
`OCTOPUS` adheres to a code of conduct adapted from the [Contributor Covenant] code of conduct.
Contributing guidelines can be found [here][contrib-guidelines].

## References
1. Maeda, A., Sano, K. and Yokoyama, T. (1988), Reconstruction by weighted correlation for MRI with time-varying gradients. IEEE Transactions on Medical Imaging, 7(1): 26-31. doi: 10.1109/42.3926
2. Noll, D. C., Pauly, J. M., Meyer, C. H., Nishimura, D. G. and Macovskj, A. (1992), Deblurring for non‐2D fourier transform magnetic resonance imaging. Magn. Reson. Med., 25: 319-333. doi:10.1002/mrm.1910250210
3. Man, L., Pauly, J. M. and Macovski, A. (1997), Multifrequency interpolation for fast off‐resonance correction. Magn. Reson. Med., 37: 785-792. doi:10.1002/mrm.1910370523
4. Noll, D. C., Meyer, C. H., Pauly, J. M., Nishimura, D. G. and Macovski, A. (1991), A homogeneity correction method for magnetic resonance imaging with time-varying gradients. IEEE Transactions on Medical Imaging, 10(4): 629-637. doi: 10.1109/42.108599
5. Schomberg, H. (1999), Off-resonance correction of MR images. IEEE Transactions on Medical Imaging, 18( 6): 481-495. doi: 10.1109/42.781014

[api-docs]: https://pypulseq.readthedocs.io/en/latest
[Bruker]: https://github.com/pulseq/bruker_interpreter
[Contributor Covenant]: http://contributor-covenant.org
[contrib-guidelines]: https://github.com/imr-framework/OCTOPUS/blob/master/CONTRIBUTING.md
[GE]: https://toppemri.github.io
[google-colab]: https://colab.research.google.com/
[installation]: #installation
[lightning-start]: #lightning-start----pypulseq-in-your-browser
[notebook-examples]: https://github.com/imr-framework/pypulseq/tree/master/pypulseq/seq_examples/notebooks
[Pulseq specification]: https://pulseq.github.io/specification.pdf
[scholar-citations]: https://scholar.google.com/scholar?oi=bibs&hl=en&cites=16703093871665262997
[script-examples]: https://github.com/imr-framework/pypulseq/tree/master/pypulseq/seq_examples/script