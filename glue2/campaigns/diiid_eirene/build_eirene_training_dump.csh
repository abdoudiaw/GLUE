#!/bin/tcsh -f
# SOLPS-ITER--SOLSTICE neutral-source coupling project
# Project author and maintainer: Abdou Diaw

set source_dir = /home/cloud/local/solps/solps-iter-eirene-training-dump
if ($#argv >= 1) then
  set source_dir = $argv[1]
endif
set build_mode = serial
if ($#argv >= 2) then
  set build_mode = $argv[2]
endif
if ("$build_mode" != "serial" && "$build_mode" != "mpi") then
  echo "Usage: $0 [source-dir] [serial|mpi]"
  exit 2
endif

setenv SOLPS_HOST_NAME_FORCE ORNL
setenv NO_MANUAL true
setenv NO_CMAKE true

cd $source_dir
source setup.csh gfortran

cd modules/B2.5
make VERSION
if ($status != 0) exit $status

if ("$build_mode" == "mpi") then
  make MPI_FC=/usr/bin/mpif90 USE_EIRENE=-DB25_EIRENE \
    USE_MPI=-DUSE_MPI SOLPS_MPI=yes \
    $source_dir/modules/B2.5/builds/couple_SOLPS-ITER.ORNL.gfortran.mpi/b2mod_dimensions.mod
else
  make USE_EIRENE=-DB25_EIRENE \
    $source_dir/modules/B2.5/builds/couple_SOLPS-ITER.ORNL.gfortran/b2mod_dimensions.mod
endif
if ($status != 0) exit $status

cd ../Eirene
make -f config/Makefile VERSION
if ($status != 0) exit $status

cd ../..
if ("$build_mode" == "mpi") then
  make MAKE_OPTIONS=-j8 MPI_FC=/usr/bin/mpif90 \
    SOLPS_CPP="-DGFORTRAN -DNCAR4 -DNO_NAG -DNO_JSON" b25eirene_nox_mpi
  set build_status = $status
else
  make MAKE_OPTIONS=-j8 \
    SOLPS_CPP="-DGFORTRAN -DNCAR4 -DNO_NAG -DNO_JSON" b25eirene_nox
  set build_status = $status
endif
exit $build_status
