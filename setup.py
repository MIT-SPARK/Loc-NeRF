from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

args = generate_distutils_setup(
    packages=['locnerf'],
    package_dir={'':'src'}
)
setup(**args)