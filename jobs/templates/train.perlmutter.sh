#!/bin/bash
# Training job on Perlmutter: rendered here, submitted by the user from $SCRATCH/GLUE.
#   ${submit_hint}
cd ${glue_root} && git pull && git submodule update --init external/solstice
export PYTHONPATH=$PWD:$PWD/external/solstice/src
${commands}
