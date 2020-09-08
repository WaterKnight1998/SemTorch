# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open

# For installing PyTorch and Torchvision in Windows
import sys
import subprocess

# Get the long description from the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements("requirements.txt")

def remove_requirements(requirements, remove_elem):
    new_requirements = []
    for requirement in requirements:
        if remove_elem not in requirement:
            new_requirements.append(requirement)

    return new_requirements

version = None
with open('semtorch/__init__.py', 'r', encoding='utf-8') as f:

    version = f.readline().split('=')[-1].strip().replace('"','')



# Windows specific requirements
print(f"Platform: {sys.platform}")
if sys.platform in ['win32','cygwin','windows']:
    torch_version = "torch>=1.6.0,<2.0.0"
    torchvision_version = "torchvision>=0.7.0,<1.0.0"

    for requirement in install_reqs:
        if "torch" in requirement:
            torch_version = requirement
        if "torchvision" in requirement:
            torchvision_version = requirement
    
    install_reqs = remove_requirements(install_reqs,'torch')
    install_reqs = remove_requirements(install_reqs,'torchvision')

    print('Trying to install PyTorch and Torchvision!')
    code = 1
    try:
        code = subprocess.call(['pip', 'install', torch_version, torchvision_version, '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        if code != 0:
            raise Exception('PyTorch and Torchvision installation failed !')
    except:
        try:
            code = subprocess.call(['pip3', 'install', torch_version, torchvision_version, '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
            if code != 0:
                raise Exception('PyTorch and Torchvision installation failed !')
        except:
            print('Failed to install PyTorch, please install PyTorch and Torchvision manually following the simple instructions at: https://pytorch.org/get-started/')
    if code == 0:
        print('Successfully installed PyTorch and torchvision! (If you need the GPU version, please install it manually, checkout the mindsdb docs and the pytorch docs if you need help)')


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='SemTorch',  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,  # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='Deep Leaarning segmentation architectures for PyTorch and FastAI',  # Required

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",

    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='https://github.com/WaterKnight1998/SemTorch',  # Optional

    # This should be your name or the name of the organization which owns the
    # project.
    author='David Lacalle Castillo',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='dvdlacallecastillo@gmail.com',  # Optional

    maintainer='David Lacalle Castillo',
    maintainer_email='dvdlacallecastillo@gmail.com',

    license="Apache License Version 2.0",
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',       

        # Topics
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',


        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='instance semantic segmentation pytorch fastai fastai2 fastaiv2 saliend object detection',  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(),  # Required

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=install_reqs,  # Optional

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    #extras_require={  # Optional
    #    'dev': ['check-manifest'],
    #    'test': ['coverage'],
    #},

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #
    # If using Python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    include_package_data=True,
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #data_files=[('my_data', ['data/data_file'])],  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
)
