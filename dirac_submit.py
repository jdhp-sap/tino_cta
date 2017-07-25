#!/usr/bin/env python2

import glob
import sys
import re

from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac


def sliding_window(my_list, window_size, step_size=None, start=0):
    if step_size is None:
        step_size = window_size
    start = start
    while start+window_size < len(my_list):
        yield my_list[start:start+window_size]
        start += step_size
    else:
        yield my_list[start:]


# this thing submits all the jobs
dirac = Dirac()

#   ######  ######## ######## ######## ########  #### ##    ##  ######
#  ##    ##    ##    ##       ##       ##     ##  ##  ###   ## ##    ##
#  ##          ##    ##       ##       ##     ##  ##  ####  ## ##
#   ######     ##    ######   ######   ########   ##  ## ## ## ##   ####
#        ##    ##    ##       ##       ##   ##    ##  ##  #### ##    ##
#  ##    ##    ##    ##       ##       ##    ##   ##  ##   ### ##    ##
#   ######     ##    ######## ######## ##     ## #### ##    ##  ######
wavelet_args = "-K -C1 -m3 -s2,2,3 -n4"

pilot_args = ' '.join((
        'classify_and_reconstruct.py',
        '--classifier ./{classifier}',
        '--out_file {out_file}',
        '--store',
        '--indir ./ --infile_list run*.gz',
        '--tail',  # comment out to do wavelet cleaning
        ))


# files containing lists of the ASTRI files on the GRID
astri_filelist_gamma = open("/local/home/tmichael/Data/cta/ASTRI9/vo.cta.in2p3.fr"
                            "-user-c-ciro.bigongiari-MiniArray9-Simtel-gamma.lfns")
astri_filelist_proton = open("/local/home/tmichael/Data/cta/ASTRI9/vo.cta.in2p3.fr"
                             "-user-c-ciro.bigongiari-MiniArray9-Simtel-proton.lfns")

# set here the `run_token` of files you want to resubmit
# submits everything if empty
redo = []

# proton files are smaller, can afford more files per run -- at a ratio 11:3
window_sizes = [3*4, 11*4]
# I used the first few files to train the classifier -- skip these
start_runs = [14, 100]

mode = "tail" if "tail" in pilot_args else "wave"

# the pickled classifier on the GRID
classifier_LFN = "LFN:/vo.cta.in2p3.fr/user/t/tmichael/cta/meta/classifier/"\
                 "{}/classifier_{}_{}_{}.pkl".format(
                         "ASTRI", mode,
                         re.sub('[ ,]', '', wavelet_args),
                         "RandomForestClassifier")


# define a template name for the file that's going to be written out.
# the placeholder braces are going to get set during the file-loop
output_filename_template = 'classified_events_{}_{}_{}.h5'
output_path = "cta/astri_mini/"

# sets all the local files that are going to be uploaded with the job plus the pickled
# classifier (if the file name starts with `LFN:`, it will be copied from the GRID itself)
input_sandbox = ['modules', 'helper_functions.py',

                 # python wrapper for the mr_filter wavelet cleaning
                 '/local/home/tmichael/software/jeremie_cta/'
                 'sap-cta-data-pipeline/datapipe/',

                 # sets up the environment + script that is being run
                 'dirac_pilot.sh', pilot_args.split()[0],

                 # the pickled model for the event classifier
                 classifier_LFN,

                 # the executable for the wavelet cleaning
                 'LFN:/vo.cta.in2p3.fr/user/t/tmichael/cta/bin/mr_filter/v3_1/mr_filter',
                 ]


print("\nrunning as:")
print(pilot_args)
print("\nwith input_sandbox:")
print(input_sandbox)
print("\nwith output file:")
print(output_filename_template.format(
        '{particle-type}', '{cleaning-mode}', '{run-token}'))


for i, astri_filelist in enumerate([astri_filelist_gamma, astri_filelist_proton]):
    window_size = window_sizes[i]
    start_run = start_runs[i]
    for run_filelist in sliding_window([l.strip() for l in astri_filelist],
                                       window_size, start=start_run):

        channel = "gamma" if "gamma" in " ".join(run_filelist) else "proton"

        # this selects the `runxxx` part of the first and last file in the run
        # list and joins them with a dash so that we get a nice identifier in
        # the outpult file name. if there is only one file in the list, use only that one
        run_token = re.split('/|\.', run_filelist[+0])[-3]
        if len(run_filelist) > 1:
            run_token = '-'.join([run_token, re.split('/|\.', run_filelist[-1])[-3]])

        # if some jobs failed and you want to resubmit, add their token to `redo`
        if redo and run_token not in redo:
            continue

        print("\nrunning on {} file{}:".format(len(run_filelist),
                                               "" if len(run_filelist) == 1 else "s"))
        print(run_filelist)

        j = Job()
        j.setCPUTime(30000)  # 1 h in seconds times 8 (CPU normalisation factor)
        j.setName('classifier {}.{}.{}'.format(channel, mode, run_token))

        # bad sites -- here miniconda cannot be found (due to bad vo configuration?)
        j.setBannedSites(['LCG.CAMK.pl'])

        j.setInputSandbox(input_sandbox +
                          # adding the data files into the input sandbox instead of input
                          # data to not be bound to the very busy frascati site
                          [f.replace('/vo', 'LFN:/vo') for f in run_filelist])

        # setting input
        # j.setInputData(run_filelist)

        # setting output
        output_filename = output_filename_template.format(channel, mode, run_token)

        # print("\nOutputSandbox: {}".format(output_filename))
        # j.setOutputSandbox([output_filename])

        print("\nOutputData: {}{}".format(output_path, output_filename))
        j.setOutputData([output_filename], outputSE=None, outputPath=output_path)

        # the `dirac_pilot.sh` is the executable. it sets up the environment and then
        # starts the script with all parameters given by `pilot_args`
        j.setExecutable('dirac_pilot.sh',
                        pilot_args.format(out_file=output_filename,
                                          classifier=classifier_LFN.split('/')[-1]))

        # check if we should somehow stop doing what we are doing
        if "dry" in sys.argv:
            print("\nrunning dry -- not submitting")
            break

        # this sends the job to the GRID and uploads all the files into the input sandbox
        # in the process
        print("\nsubmitting job")
        res = dirac.submit(j)
        print('Submission Result: {}'.format(res['Value']))

        if "test" in sys.argv:
            print("test run -- only submitting one job")
            break

    # since there are two nested loops, need to break again
    if "dry" in sys.argv:
        break


print("all done -- exiting now")
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
