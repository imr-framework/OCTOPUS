import setuptools

try:  # Unicode decode error on Windows
    with open("README.md", "r") as fh:
        long_description = fh.read()
except:
    long_description = 'Off-resonance correction of MR images'

with open('requirements.txt', 'r') as f:
    install_reqs = f.read().strip()
    install_reqs = install_reqs.split("\n")

setuptools.setup(
    name = 'MR-OCTOPUS',         # How you named your package folder (MyLib)   # Chose the same as "name"
    version = '0.1.0',      # Start with a small number and increase it with every change you make
    author = 'Marina Manso Jimeno',                   # Type in your name
    author_email = 'mm5290@columbia.edu',      # Type in your E-Mail
    description = 'Off-resonance correction of MR images',   # Give a short description about your library
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = 'https://github.com/imr-framework/OCTOPUS',   # Provide either the link to your github or to your website
    packages = setuptools.find_packages(),
    include_package_data = True,
    install_requires = install_reqs,
    license = 'License :: OSI Approved :: GNU Affero General Public License v3',
    classifiers = [
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
)