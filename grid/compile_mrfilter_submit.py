from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Dirac import Dirac
from DIRAC.Interfaces.API.Job import Job


dirac = Dirac()


j = Job()
j.setName("compile_mrfilter")
j.setCPUTime(80)
j.setInputSandbox(["dirac_compile_mrfilter_pilot.sh"])
j.setExecutable("dirac_compile_mrfilter_pilot.sh", "")
j.setOutputData(["mr_filter"], outputSE=None,
                outputPath="cta/bin/mr_filter/v3_1/")
Dirac().submit(j)
