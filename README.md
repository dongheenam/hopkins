# hopkins.py

This simple Python program regenerates the last-crossing Hopkins IMF by solving the Volterra equation as described in Hopkins (2012MNRAS.423.2016H), (2012MNRAS.423.2037H), and (2013MNRAS.430.1653H).

- Required Python version: 3.6 (simply replace all f-strings and it should also be compatible with earlier versions of Python 3)
- Required modules: numPy, sciPy, h5py (only used for writing the IMF; replace lines 213-231 with open() or pickle to remove h5py dependency)
