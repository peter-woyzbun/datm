from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import sys


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        print "JUST INSTALLED THE DATM PACKAGE IN DEVELOPMENT MODE!!!!"
        sys.stdout.write("JUST INSTALLED THE DATM PACKAGE IN DEVELOPMENT MODE!!!!")
        develop.run(self)
        print "JUST INSTALLED THE DATM PACKAGE IN DEVELOPMENT MODE!!!!"


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        print "JUST INSTALLED THE DATM PACKAGE!!!"
        install.run(self)
        print "JUST INSTALLED THE DATM PACKAGE IN DEVELOPMENT MODE!!!!"

setup(name='datm',
      version='0.1',
      description='DATm package.',
      url='http://github.com/peter-woyzbun/',
      author='Peter Woyzbun',
      author_email='peter.woyzbun@gmail.com',
      license='MIT',
      cmdclass={
              'develop': PostDevelopCommand,
              'install': PostInstallCommand,
          },
      packages=['datm'],
      install_requires=['click', 'pandas', 'django', 'numpy', 'pandasql'],
      entry_points={
        'console_scripts': ['datm=datm.management.cmd_line:cmd_group'],
      },
      zip_safe=False)