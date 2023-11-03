import setuptools

setuptools.setup(
   name='torch_erf',
   version='0.99',
   description='Torch module implementation of the complex error function based on Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518',
   author='Nicolo Rossi',
   author_email='nicolo.rossi@bsse.ethz.ch',
   install_requires=['wheel', 'torch'],
   packages=setuptools.find_packages()
)
