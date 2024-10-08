"""
Calculate data weights based on the standard deviation within each spw using
`statwt`.
"""

from casatasks import statwt

from . import pipeline_save
from .utils import logprint, runtiming


def task_logprint(msg):
    logprint(msg, logfileout="logs/statwt.log")


task_logprint("*** Starting EVLA_pipe_statwt.py ***")
time_list = runtiming("checkflag", "start")
QA2_statwt = "Pass"

task_logprint("Calculating data weights per SpW using statwt.")

# FIXME Should check for a `cont.dat` file here and feed to `fitspw` and
# `field` parameters of `statwt`.

# Run on all calibrators
statwt(
    vis=ms_active,
    minsamp=2,
    intent="*CALIBRATE*",
    datacolumn="corrected",
)

# Run on all targets
# set spw to exclude strong science spectral lines
statwt(
    vis=ms_active,
    minsamp=2,
    intent="*TARGET*",
    datacolumn="corrected",
)

# Until we understand better the failure modes of this task, leave QA2
# score set to "Pass".

task_logprint(f"QA2 score: {QA2_statwt}")
task_logprint("Finished EVLA_pipe_statwt.py")
time_list = runtiming("statwt", "end")

pipeline_save()
