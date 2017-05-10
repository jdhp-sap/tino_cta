 #!/bin/bash

echo "pwd:"
pwd

echo "ls -lhrt"
ls -lhrt

# setting some paths
export PATH=./:$PATH
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH

# sourcing the ctapipe miniconda environment
export MINICONDA=/cvmfs/cta.in2p3.fr/software/miniconda
export PATH=$MINICONDA/bin:$PATH
source $MINICONDA/bin/activate ctapipe_v0.4


echo '$PATH'
echo $PATH
echo '$LD_LIBRARY_PATH'
echo $LD_LIBRARY_PATH
echo '$PYTHONPATH'
echo $PYTHONPATH

echo 'which python'
which python

# prevent matplotlib to complain about missing backends
export MPLBACKEND=Agg

MYHOME=$PWD

# getting and compiling cfitsio on site
{
    if [ ! -e cfitsio_latest.tar.gz ]
    then
        echo getting cfitsio
        wget http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz \
        &> /dev/null
    else
        echo cfitsio tarball already here
    fi
    tar -xzvf cfitsio_latest.tar.gz
    cd cfitsio
    ./configure
    make
    make install

    export CFITSIO=$PWD
    export LD_LIBRARY_PATH=$CFITSIO:$LD_LIBRARY_PATH

    cd $MYHOME
} &> /dev/null

# get and compile mr_filter
{
    if [ ! -e ISAP*tgz ]
    then
        echo getting ISAP
        wget http://www.cosmostat.org/wp-content/uploads/2014/12/ISAP_V3.1.tgz \
        &> /dev/null
    else
        echo ISAP tarball already here
    fi
    tar -xzf ISAP_V3.1.tgz
    export ISAP=$PWD/ISAP_V3.1

    cd $ISAP/cxx

    echo
    echo Line ${LINENO}
    pwd
    ls

    # sparse2d looks in extern/cfitsio for libraries and headers
    # add the directories and link to cfitsio
    mkdir -p extern/cfitsio
    ln -s $CFITSIO $PWD/extern/cfitsio/include
    ln -s $CFITSIO/libcfitsio.a $PWD/extern/cfitsio

    tar -xzf sparse2d_V1.1.tgz
    cd sparse2d

    ./configure
    make

    cd $MYHOME
}

cp $ISAP/cxx/sparse2d/bin/mr_filter $MYHOME

# executing the script that we intend to run
echo calling: 'python $@'
python $@


echo
echo "final ls -lhrt"
ls -lhrt
