#!/usr/bin/env python2

import os
from os.path import basename, expandvars
import sys
import glob
import re
import random
import datetime

from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac


def sliding_window(my_list, window_size, step_size=None, start=0):
    step_size = step_size or window_size
    while start + window_size < len(my_list):
        yield my_list[start:start + window_size]
        start += step_size
    else:
        yield my_list[start:]


# this thing submits all the jobs
dirac = Dirac()


# list of files on my GRID SE space
# not submitting jobs where we already have the output
while True:
    try:
        GRID_filelist = open('vo.cta.in2p3.fr-user-t-tmichael.lfns').read()
        break
    except IOError:
        os.system('dirac-dms-user-lfns')


#   ######  ######## ######## ######## ########  #### ##    ##  ######
#  ##    ##    ##    ##       ##       ##     ##  ##  ###   ## ##    ##
#  ##          ##    ##       ##       ##     ##  ##  ####  ## ##
#   ######     ##    ######   ######   ########   ##  ## ## ## ##   ####
#        ##    ##    ##       ##       ##   ##    ##  ##  #### ##    ##
#  ##    ##    ##    ##       ##       ##    ##   ##  ##   ### ##    ##
#   ######     ##    ######## ######## ##     ## #### ##    ##  ######

# bad sites
banned_sites = [
    # 'LCG.CPPM.fr',   # LFN connection problems
    # 'LCG.IN2P3-CC.fr',  # jobs fail immediately after start
    # 'LCG.CAMK.pl',      # no miniconda (bad vo configuration?)
    # 'LCG.Prague.cz',    # no miniconda (bad vo configuration?)
    # 'LCG.PRAGUE-CESNET.cz',    # no miniconda (bad vo configuration?)
    'LCG.OBSPM.fr',
    # 'LCG.LAPP.fr',      # no miniconda (bad vo configuration?)
    # 'LCG.PIC.es',       #
    # 'LCG.M3PEC.fr',     #
    # 'LCG.CETA.es'
]

cam_id_list = ["LSTCam", "NectarCam", "DigiCam"]

source_ctapipe = \
    'source /cvmfs/cta.in2p3.fr/software/miniconda/bin/activate ctapipe_v0.5.3'
execute = './classify_and_reconstruct.py'
pilot_args_classify = ' '.join([
        source_ctapipe, '&&',
        execute,
        '--classifier ./{classifier}',
        '--regressor ./{regressor}',
        '--outfile {outfile}',
        '--indir ./ --infile_list *.simtel.gz',
        '--{mode}',
        '--cam_ids'] + cam_id_list)
pilot_args_append = ' '.join([
        source_ctapipe, '&&',
        './append_tables.py',
        '--infiles_base', '{in_name}',
        '--outfile', '{out_name}'])

expandvars
# files containing lists of the Prod3b files on the GRID
prod3b_filelist_gamma = open(expandvars("$CTA_DATA/Prod3b/Paranal/"
                             "Paranal_gamma_North_20deg_HB9_merged.list"))
prod3b_filelist_proton = open(expandvars("$CTA_DATA/Prod3b/Paranal/"
                              "Paranal_proton_North_20deg_HB9_merged.list"))
prod3b_filelist_electron = open(expandvars("$CTA_DATA/Prod3b/Paranal/"
                                "Paranal_electron_North_20deg_HB9_merged.list"))


# number of files per job
window_sizes = [25] * 3

# I used the first few files to train the classifier and regressor -- skip them
start_runs = [50, 50, 0]

# how many jobs to submit at once
NJobs = 200  # put at < 0 to deactivate

# define a template name for the file that's going to be written out.
# the placeholder braces are going to get set during the file-loop
output_filename_template = 'classified_events_{}.h5'
output_path = "cta/prod3b/paranal_LND_edge/"

# sets all the local files that are going to be uploaded with the job plus the pickled
# classifier (if the file name starts with `LFN:`, it will be copied from the GRID itself)
input_sandbox = [expandvars('$CTA_SOFT/tino_cta/tino_cta'),
                 expandvars('$CTA_SOFT/tino_cta/helper_functions.py'),
                 expandvars('$CTA_SOFT/tino_cta/snippets/append_tables.py'),

                 # python wrapper for the mr_filter wavelet cleaning
                 expandvars('$CTA_SOFT/jeremie_cta/ctapipe-wavelet-cleaning/datapipe/'),

                 # script that is being run
                 expandvars('$CTA_SOFT/tino_cta/' + execute), 'pilot.sh',

                 # the executable for the wavelet cleaning
                 'LFN:/vo.cta.in2p3.fr/user/t/tmichael/cta/bin/mr_filter/v3_1/mr_filter'
                 ]

# the pickled classifier and regressor on the GRID
model_path_template = \
    "LFN:/vo.cta.in2p3.fr/user/t/tmichael/cta/meta/ml_models/{}/{}_{}_{}_{}.pkl"
for cam_id in cam_id_list:
    for mode in ["wave", "tail"]:
        for model in [("regressor", "RandomForestRegressor"),
                      ("classifier", "RandomForestClassifier")]:
            input_sandbox.append(
                model_path_template.format(
                    "prod3b/paranal_edge",
                    model[0],
                    mode,
                    cam_id,
                    model[1])
            )

#
# ########  ##     ## ##    ## ##    ## #### ##    ##  ######
# ##     ## ##     ## ###   ## ###   ##  ##  ###   ## ##    ##
# ##     ## ##     ## ####  ## ####  ##  ##  ####  ## ##
# ########  ##     ## ## ## ## ## ## ##  ##  ## ## ## ##   ####
# ##   ##   ##     ## ##  #### ##  ####  ##  ##  #### ##    ##
# ##    ##  ##     ## ##   ### ##   ###  ##  ##   ### ##    ##
# ##     ##  #######  ##    ## ##    ## #### ##    ##  ######

# get jobs from today and yesterday...
days = []
for i in range(2):  # how many days do you want to look back?
    days.append((datetime.date.today() - datetime.timedelta(days=i)).isoformat())

# get list of run_tokens that are currently running / waiting
running_ids = set()
running_names = []
for status in ["Waiting", "Running", "Checking"]:
    for day in days:
        try:
            [running_ids.add(id) for id in dirac.selectJobs(
                status=status, date=day,
                owner="tmichael")['Value']]
        except KeyError:
            pass

n_jobs = len(running_ids)
if n_jobs > 0:
    print("getting names from {} running/waiting jobs... please wait..."
          .format(n_jobs))
    for i, id in enumerate(running_ids):
        if ((100 * i) / n_jobs) % 5 == 0:
            print("\r{} %".format(((20 * i) / n_jobs) * 5)),
        jobname = dirac.attributes(id)["Value"]["JobName"]
        running_names.append(jobname)
    else:
        print("\n... done")

# summary before submitting
print("\nrunning as:")
print(pilot_args_classify)
print("\nwith input_sandbox:")
print(input_sandbox)
print("\nwith output file:")
print(output_filename_template.format('{job_name}'))

for i, filelist in enumerate([
        prod3b_filelist_gamma,
        prod3b_filelist_proton,
        prod3b_filelist_electron]):
    for run_filelist in sliding_window([l.strip() for l in filelist],
                                       window_sizes[i], start=start_runs[i]):

        if "gamma" in " ".join(run_filelist):
            channel = "gamma"
        elif "proton" in " ".join(run_filelist):
            channel = "proton"
        elif "electron" in " ".join(run_filelist):
            channel = "electron"
        else:
            print("not a known channel ... skipping this filelist:")
            break

        # this selects the `runxxx` part of the first and last file in the run
        # list and joins them with a dash so that we get a nice identifier in
        # the outpult file name. if there is only one file in the list, use only that one
        run_token = re.split('_', run_filelist[+0])[3]
        if len(run_filelist) > 1:
            run_token = '-'.join([run_token, re.split('_', run_filelist[-1])[3]])

        print("-" * 20)

        # setting output name
        job_name = '_'.join([channel, run_token])
        output_filename_wave = output_filename_template.format(
            '_'.join(["wave", channel, run_token]))
        output_filename_tail = output_filename_template.format(
            '_'.join(["tail", channel, run_token]))

        # if job already running / waiting, skip
        if job_name in running_names:
            print("\n{} still running\n".format(job_name))
            continue

        # if file already in GRID storage, skip
        # (you cannot overwrite it there, delete it and resubmit)
        # (assumes tail and wave will always be written out together)
        if output_filename_wave in GRID_filelist:
            print("\n{} already on GRID SE\n".format(job_name))
            continue

        if NJobs == 0:
            print("maximum number of jobs to submit reached")
            print("breaking loop now")
            break
        else:
            NJobs -= 1

        j = Job()
        # runtime in seconds times 8 (CPU normalisation factor)
        j.setCPUTime(6 * 3600 * 8)
        j.setName(job_name)
        j.setInputSandbox(input_sandbox)

        if banned_sites:
            j.setBannedSites(banned_sites)

        # mr_filter loses its executable property by uploading it to the GRID SE; reset
        j.setExecutable('chmod', '+x mr_filter')

        j.setExecutable('ls -lah')

        for run_file in run_filelist:
            file_token = re.split('_', run_file)[3]

            # wait for a random number of seconds (up to five minutes) before starting
            # to add a bit more entropy in the starting times of the dirac querries.
            # if too many jobs try in parallel to access the SEs, the interface crashes
            sleep = random.randint(0, 5 * 60)
            j.setExecutable('sleep', str(sleep))

            # consecutively downloads the data files, processes them, deletes the input
            # and goes on to the next input file; afterwards, the output files are merged
            j.setExecutable('dirac-dms-get-file', "LFN:" + run_file)

            # consecutively process file with wavelet and tailcut cleaning
            for mode in ["wave", "tail"]:
                # source the miniconda ctapipe environment and run the python script with
                # all its arguments
                output_filename_temp = output_filename_template.format(
                    "_".join([mode, channel, file_token]))
                classifier_LFN = model_path_template.format(
                    "prod3b/paranal_edge", "classifier",
                    mode, "{cam_id}", "RandomForestClassifier")
                regressor_LFN = model_path_template.format(
                    "prod3b/paranal_edge", "regressor",
                    mode, "{cam_id}", "RandomForestRegressor")
                j.setExecutable('./pilot.sh',
                                pilot_args_classify.format(
                                    outfile=output_filename_temp,
                                    regressor=basename(regressor_LFN),
                                    classifier=basename(classifier_LFN),
                                    mode=mode))

            # remove the current file to clear space
            j.setExecutable('rm', basename(run_file))

        # simple `ls` for good measure
        j.setExecutable('ls', '-lh')

        # if there is more than one file per job, merge the output tables
        if window_sizes[i] > 1:
            for in_name, out_name in [('classified_events_wave', output_filename_wave),
                                      ('classified_events_tail', output_filename_tail)]:
                j.setExecutable('./pilot.sh',
                                pilot_args_append.format(
                                    in_name=in_name,
                                    out_name=out_name))

        print
        print("OutputData: {}{}".format(output_path, output_filename_wave))
        print("OutputData: {}{}".format(output_path, output_filename_tail))
        j.setOutputData([output_filename_wave, output_filename_tail],
                        outputSE=None, outputPath=output_path)

        # check if we should somehow stop doing what we are doing
        if "dry" in sys.argv:
            print("\nrunning dry -- not submitting")
            break

        # this sends the job to the GRID and uploads all the
        # files into the input sandbox in the process
        print("\nsubmitting job")
        print('Submission Result: {}\n'.format(dirac.submit(j)['Value']))

        # break if this is only a test submission
        if "test" in sys.argv:
            print("test run -- only submitting one job")
            break

    # since there are two nested loops, need to break again
    if "dry" in sys.argv or "test" in sys.argv:
        break

try:
    os.remove("datapipe.tar.gz")
    os.remove("modules.tar.gz")
except:
    pass

print("\nall done -- exiting now")
exit()

# ########    ###     #######
# ##         ## ##   ##     ##
# ##        ##   ##  ##     ##
# ######   ##     ## ##     ##
# ##       ######### ##  ## ##
# ##       ##     ## ##    ##
# ##       ##     ##  ##### ##

# specify allowed sites to send the job to
j.setDestination(['LCG.IN2P3-CC.fr', 'LCG.DESY-ZEUTHEN.de', 'LCG.CNAF.it',
                  'LCG.GRIF.fr', 'LCG.CYFRONET.pl', 'LCG.PRAGUE-CESNET.cz',
                  'LCG.Prague.cz', 'LCG.LAPP.fr', 'LCG.CIEMAT.es', 'LCG.PIC.es'])

# to specify input GRID files independent of the site the job is send to
file1 = 'LFN:/vo.cta.in2p3.fr/user/c/ciro.bigongiari/MiniArray9/Simtel/'\
        'gamma/run1011.simtel.gz'
j.setInputSandbox([file1])
