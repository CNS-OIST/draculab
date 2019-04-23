# __init__.py
# Used to import all files in the directory.
# Based on:
# https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder

from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
