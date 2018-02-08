#!/usr/bin/env python2

from os.path import basename, expandvars
from glob import glob

from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Dirac import Dirac

dirac = Dirac()

file_collection = []
for line in open(expandvars('$CTA_SOFT/tino_cta/vo.cta.in2p3.fr-user-t-tmichael.lfns')):
    line = line.strip()

    if "prod3b/paranal_LND/classified_events_proton_tail" not in line:
        continue

    # don't download if already in current directory
    if glob(basename(line)):
        continue

    if len(file_collection) < 100:
        file_collection.append(line)
    else:
        dirac.getFile(file_collection)
        file_collection = []

if file_collection:
    dirac.getFile(file_collection)
