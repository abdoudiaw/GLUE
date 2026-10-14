#!/bin/tcsh
# Rebuild the coupled B2.5-EIRENE executable of the training-dump fork on Mora.
# make goes by mtime: the files listed in `touch` are touched first (a source put back with an
# older mtime than its object is silently not recompiled), and the symbols listed in `expect`
# are checked in the result.
cd ${solps_tree}
${launch_env_tcsh}
foreach f (${touch_files})
  touch modules/B2.5/$f
end
echo "BUILD START `date`"
make -j${make_jobs} b25eirene_mpi >& ${log_dir}/${name}.make.log
set rc = $status
echo "BUILD EXIT $rc `date`"
set exe = modules/B2.5/builds/couple_SOLPS-ITER.ORNL.gfortran.mpi/b2mn.exe
ls -la $exe
foreach s (${expect_symbols})
  echo "symbol $s: `nm $exe | grep -c -i $s`"
end
grep -i -m5 " error" ${log_dir}/${name}.make.log | grep -v "Error building b2cdc"
echo "JOB DONE `date`"
