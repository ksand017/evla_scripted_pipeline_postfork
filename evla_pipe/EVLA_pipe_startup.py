"""
The pipeline assumes all files for a single dataset are located in
the current directory, and that this directory contains only files
relating to this dataset.
"""

from . import pipeline_save
from .utils import (runtiming, logprint)


def task_logprint(msg):
    logprint(msg, logfileout="logs/startup.log")


task_logprint("*** Starting EVLA_pipe_startup.py ***")
runtiming('startup', 'start')

task_logprint(f"Running from path: {os.getcwd()}")

# File names
#
# if SDM_name is already defined, then assume it holds the SDM directory
# name, otherwise, read it in from stdin
#
SDM_name_already_defined = 1
try:
    SDM_name
except NameError:
    SDM_name_already_defined = 0
    SDM_name = input("Enter SDM file name: ")
    if SDM_name == "":
        raise RuntimeError("SDM name must be given.")

# Trap for '.ms', just in case, also for directory slash if present:
SDM_name = SDM_name.rstrip('/')
if SDM_name.endswith('.ms'):
    SDM_name = SDM_name[:-3]
msname = f"{SDM_name}.ms"

# FIXME This is terribly non-robust.  should really trap all the inputs from
# the automatic pipeline (the root directory and the relative paths).  and also
# make sure that 'rawdata' only occurs once in the string.  but for now, take
# the quick and easy route.
if SDM_name_already_defined:
    msname = msname.replace('rawdata', 'working')

if not os.path.isdir(msname):
    while not os.path.isdir(SDM_name) and not os.path.isdir(msname):
        print(f"{SDM_name} is not a valid SDM directory")
        SDM_name = input("Re-enter a valid SDM directory (without '.ms'): ")
        SDM_name = SDM_name.rstrip('/')
        if SDM_name.endswith('.ms'):
            SDM_name = SDM_name[:-3]
        msname = f"{SDM_name}.ms"

mshsmooth = f"{SDM_name}.hsmooth.ms"
if SDM_name_already_defined:
    mshsmooth = mshsmooth.replace('rawdata', 'working')
ms_spave = f"{SDM_name}.spave.ms"
if SDM_name_already_defined:
    ms_spave = ms_spave.replace('rawdata', 'working')

task_logprint(f"SDM used is: {SDM_name}")

# Other inputs:

# Ask if a a real model column should be created, or the virtual model should
# be used.
mymodel_already_set = 1
try:
    mymodel
    scratch = mymodel == "y"
except NameError:
    mymodel_already_set = 0
    mymodel = input("Create the real model column (y/[n]): ").lower()
    mymodel = "n" if mymodel != "y" else mymodel
    scratch = mymodel == "y"

myHanning_already_set = 1
try:
    #myHanning
    do_hanning
except NameError:
    myHanning_already_set = 0
    hanning_input_results = input("Hanning smooth the data (y/[n]): ").lower()
    do_hanning = hanning_input_results not in ("", "n")

myPol_already_set = 1
try:
    #myPol
    do_pol
except NameError:
    myPol_already_set = 0
    dopol_input_results = input("Perform polarization calibration? (y/[n]): ").lower()
    do_pol = dopol_input_results not in ("", "n")

ms_active = msname

# And ask for auxiliary information.
try:
    projectCode
except NameError:
    projectCode = 'Unknown'
try:
    piName
except NameError:
    piName = 'Unknown'
try:
    piGlobalId
except NameError:
    piGlobalId = 'Unknown'
try:
    observeDateString
except NameError:
    observeDateString = 'Unknown'
try:
    pipelineDateString
except NameError:
    pipelineDateString = 'Unknown'


# For now, use same ms name for Hanning smoothed data, for speed.
# However, we only want to smooth the data the first time around, we do
# not want to do more smoothing on restarts, so note that this parameter
# is reset to "n" after doing the smoothing in EVLA_pipe_hanning.py.

task_logprint("Finished EVLA_pipe_startup.py")
runtiming('startup', 'end')

pipeline_save()

