from sys import exit
from glob import glob
import argparse

from os.path import expandvars

import numpy as np

from itertools import chain

import matplotlib.pyplot as plt

from astropy import units as u

from ctapipe.io.hessio import hessio_event_source

import sys
sys.path.append("/local/home/tmichael/software/tino_cta/")
from helper_functions import *

import signal
stop = None
def signal_handler(signal, frame):
    global stop
    if stop:
        print('you pressed Ctrl+C again -- exiting NOW')
        exit(-1)
    print('you pressed Ctrl+C!')
    print('exiting loop after this event')
    stop = True
signal.signal(signal.SIGINT, signal_handler)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max-events', type=int, default=None)
    parser.add_argument('-i', '--indir',    type=str,
                        default=expandvars("$HOME/Data/cta/ASTRI9"))
    parser.add_argument('-r', '--runnr',    type=str, default="*")
    parser.add_argument('-t', '--tel',      type=int, default=None)
    parser.add_argument('-o', '--outtoken', type=str, default=None)
    args = parser.parse_args()

    filenamelist_gamma  = glob( "{}/gamma/run*{}.simtel.gz".format(args.indir,args.runnr ))
    filenamelist_proton = [] #glob( "{}/proton/run{}.*gz".format(args.indir,args.runnr ))

    print("{}/gamma/run*{}.simtel.gz".format(args.indir, args.runnr))
    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()

    allowed_tels = set(range(10))
    #if args.tel: allowed_tels = allowed_tels & set([args.tel])

    npe = []
    adc_cal  = []
    residual = []

    for filename in chain(sorted(filenamelist_gamma)[:],
                          sorted(filenamelist_proton)[:]):
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                allowed_tels=allowed_tels,
                                max_events=args.max_events)

        for event in source:
            tels = set(event.trig.tels_with_trigger) & set(event.dl0.tels_with_data)
            for tel_id in tels:
                cal = apply_mc_calibration_ASTRI(
                                        event.dl0.tel[tel_id].adc_sums,
                                        event.mc.tel[tel_id].dc_to_pe,
                                        event.mc.tel[tel_id].pedestal)
                print(np.sum(event.dl0.tel[tel_id].adc_sums[0]))
                print(np.sum(event.dl0.tel[tel_id].adc_sums[1]))
                print()
                MoCa = event.mc.tel[tel_id].photo_electron_image

                for i, pe in enumerate(MoCa):
                    if pe > 0:
                        npe.append(pe)
                        #if cal[i] < -500:
                            #print(pe)
                            #print(cal[i])
                            #print()
                        adc_cal .append(cal[i])
                        residual.append(cal[i] - npe)
            if stop: break
        if stop: break

    fig = plt.figure()

    fig.add_subplot(221)
    plt.hist(npe)

    fig.add_subplot(224)
    plt.hist(adc_cal)

    fig.add_subplot(222)
    bins = np.linspace(-5, 35, 160)
    plt.hist(residual, bins, alpha=0.5, label='corrected calibration')
    plt.legend(loc='upper right')


    plt.show()
