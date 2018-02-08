#!/usr/bin/env python2

from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Dirac import Dirac

dirac = Dirac()

file_collection = []
for line in open('vo.cta.in2p3.fr-user-t-tmichael.lfns'):
    line = line.strip()

    if "prod3b/paranal_LND/classified_events_electron_tail" not in line:
        continue

    if len(file_collection) < 100:
        file_collection.append(line)
    else:
        print("removing:")
        print(file_collection)
        dirac.removeFile(file_collection, True)
        file_collection = []

if file_collection:
    print("removing:")
    print(file_collection)
    dirac.removeFile(file_collection, True)
