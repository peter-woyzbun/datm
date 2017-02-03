from setuptools import setup


setup(name='datm',
      version='0.1',
      description='DATm package.',
      url='http://github.com/peter-woyzbun/',
      author='Peter Woyzbun',
      author_email='peter.woyzbun@gmail.com',
      license='MIT',
      packages=['datm'],
      install_requires=['click', 'pandas', 'django', 'numpy', 'pandasql', 'pyparsing', 'matplotlib', 'autopep8'],
      entry_points={
        'console_scripts': ['datm=datm.management.cmd_line:cmd_group'],
      },
      zip_safe=False)