#!/bin/tcsh -f
# Project: SOLPS-ITER/EIRENE training-data generation for SOLSTICE
# Author: Abdou Diaw, Oak Ridge National Laboratory
#
# Prepare one staged SOLPS case, execute one B2 step with EIRENE, and
# validate the raw schema-3 coupling records. This script runs on Mora.

if ( $#argv < 1 || $#argv > 2 ) then
    echo "Usage: $0 CASE_DIRECTORY [MPI_RANKS]"
    exit 2
endif

set case_dir = "$argv[1]"
set mpi_ranks = 64
if ( $#argv == 2 ) set mpi_ranks = "$argv[2]"

set solps_root = /home/cloud/local/solps/solps-iter-eirene-training-dump
set validator = ${solps_root}/modules/B2.5/src/test/validate_eirene_training_dump.py
set b2mn_exe = ${solps_root}/modules/B2.5/builds/couple_SOLPS-ITER.ORNL.gfortran.mpi/b2mn.exe
set solps_lib = /home/cloud/local/solps/solps-libs/lib
set base_run = /home/cloud/solps-runs/diii-d/baserun
set campaign_root = `dirname "$case_dir"`

# make compares timestamps only, so an executable can lag the checked-out
# source. Without the sheath save/restore the repeated EIRENE call sees
# index-mapped sheath inputs, and the event pair is not a valid repeat.
if ( `nm "$b2mn_exe" | grep -c -i sheath_input` < 3 ) then
    echo "ERROR: $b2mn_exe was built without the EIRENE sheath save/restore"
    exit 2
endif

if ( ! -d "$case_dir" ) then
    echo "ERROR: staged case does not exist: $case_dir"
    exit 2
endif
if ( ! -d "$base_run" ) then
    echo "ERROR: base run does not exist: $base_run"
    exit 2
endif
if ( ! -e "${campaign_root}/baserun" ) then
    ln -s "$base_run" "${campaign_root}/baserun"
endif
if ( ! -d "${campaign_root}/baserun" ) then
    echo "ERROR: invalid campaign baserun: ${campaign_root}/baserun"
    exit 2
endif

cd "$case_dir"

if ( ! -s b2mn.dat ) then
    echo "ERROR: b2mn.dat is missing"
    exit 20
endif

# Restart from the archived final plasma state.
set restart_source = b2fstate
if ( -s b2fstate ) then
    cp -p b2fstate b2fstati
else if ( -s b2fstati ) then
    set restart_source = b2fstati
    echo "WARNING: b2fstate is unavailable; using archived b2fstati"
else
    echo "ERROR: neither b2fstate nor b2fstati contains a restart state"
    exit 20
endif

# Keep geometry local to the staged case. Otherwise make may compare archived
# targets against a newer shared baserun geometry and regenerate b2fpardf and
# b2fstati, destroying the case-specific restart contract.
if ( ! -s b2fgmtry ) then
    if ( -s "${base_run}/b2fgmtry" ) then
        cp -p "${base_run}/b2fgmtry" .
    else
        echo "ERROR: b2fgmtry is unavailable in both case and baserun"
        exit 20
    endif
endif
if ( ! -s fort.30 ) then
    if ( -s "${base_run}/fort.30" ) then
        cp -p "${base_run}/fort.30" .
    else
        echo "ERROR: fort.30 is unavailable in both case and baserun"
        exit 20
    endif
endif
if ( ! -e b2mn.dat.original ) cp -p b2mn.dat b2mn.dat.original

# Change only the controls required for a one-step training-data restart.
sed -i \
  -e "s/^'b2mndr_ntim'.*/'b2mndr_ntim'                      '1'   # one-step EIRENE training dump/" \
  -e "s/^'b2mwti_2dwrite'.*/'b2mwti_2dwrite'               '0'   # disable optional LDRD output/" \
  b2mn.dat

grep -q "^'eirene_training_dump'" b2mn.dat
if ( $status == 0 ) then
    sed -i "s/^'eirene_training_dump'.*/'eirene_training_dump'        '1'   # Save raw EIRENE training events/" b2mn.dat
else
    sed -i "/^'eirene_repeat_first_call'/a'eirene_training_dump'        '1'   # Save raw EIRENE training events" b2mn.dat
endif

set dump_switch_count = `grep -c "^'eirene_training_dump'" b2mn.dat`
if ( "$dump_switch_count" != "1" ) then
    echo "ERROR: expected exactly one eirene_training_dump setting"
    exit 2
endif

# The archive preserves original timestamps. Mark staged derived files current
# so make does not regenerate and overwrite the case-specific restart state,
# run parameters, rates, or geometry from newer baserun input timestamps.
foreach staged_derived (b2fgmtry fort.30 b2fpardf b2frates)
    if ( -s "$staged_derived" ) touch "$staged_derived"
end
# b2fstati depends on b2fgmtry, so it must be touched last.
touch b2fstati

setenv SOLPS_HOST_NAME_FORCE ORNL
# Mora has no IPv6 loopback address. Disable hwloc GL discovery so mpirun
# does not hang while probing the nonexistent local X display on port 6000.
setenv HWLOC_COMPONENTS -gl
cd "$solps_root"
source setup.csh gfortran
if ( $?LD_LIBRARY_PATH ) then
    setenv LD_LIBRARY_PATH ${solps_lib}:${LD_LIBRARY_PATH}
else
    setenv LD_LIBRARY_PATH ${solps_lib}
endif

cd "$case_dir"

# Some archived cases omit generated B2 support files. Recreate those with
# the serial preprocessors, then retouch the archived restart so b2ai remains
# unnecessary. The MPI launch is reserved for the coupled b2mn calculation.
set generated_b2fpardf = 0
set generated_b2frates = 0
if ( ! -s b2fpardf ) then
    b2run b2ah >& eirene_training_setup_b2ah.log
    if ( $status != 0 || ! -s b2fpardf ) then
        echo "ERROR: failed to generate missing b2fpardf"
        exit 2
    endif
    set generated_b2fpardf = 1
endif
if ( ! -s b2frates ) then
    b2run b2ar >& eirene_training_setup_b2ar.log
    if ( $status != 0 || ! -s b2frates ) then
        echo "ERROR: failed to generate missing b2frates"
        exit 2
    endif
    set generated_b2frates = 1
endif
touch b2fgmtry fort.30 b2fpardf b2frates b2fstati

b2run -n -m \"mpirun -x LD_LIBRARY_PATH -np ${mpi_ranks}\" b2mn >& eirene_training_make_dryrun.log
if ( $status != 0 ) then
    echo "ERROR: b2run dry-run preflight failed"
    exit 2
endif
# b2ah and b2ar may be required when an archive omits the generated
# b2fpardf and b2frates files. They do not replace the plasma restart.
# b2ag is unsafe under the MPI wrapper, and b2ai would regenerate b2fstati.
grep -Eq 'b2ag\.exe|b2ai\.exe' eirene_training_make_dryrun.log
if ( $status == 0 ) then
    echo "ERROR: preflight would regenerate geometry or the archived b2fstati restart"
    cat eirene_training_make_dryrun.log
    exit 2
endif

b2run -m \"mpirun -x LD_LIBRARY_PATH -np ${mpi_ranks}\" b2mn >& eirene_training_run.log
set run_status = $status
if ( $run_status != 0 ) then
    echo "ERROR: b2run failed with status $run_status"
    exit $run_status
endif

# b2run returns zero when b2mn stops; a completed step prints its ITER line.
grep -Eq '^ +ITER +1 ' eirene_training_run.log
if ( $status != 0 ) then
    echo "ERROR: B2 did not complete the time step"
    exit 4
endif

set events = ( eirene_training_v3_b2call_00000000_single_call_0001.nc \
               eirene_training_v3_b2call_00000000_single_call_0002.nc )
foreach event ( $events )
    if ( ! -s $event ) then
        echo "ERROR: expected schema-3 event file was not produced: $event"
        exit 3
    endif
    # Lossless deflate; the validator below reads the files that are archived.
    nccopy -d4 -s $event ${event}.tmp
    if ( $status != 0 ) then
        echo "ERROR: failed to compress $event"
        exit 3
    endif
    mv ${event}.tmp $event
end

python3 "$validator" . >& eirene_training_validation.log
set validation_status = $status
cat eirene_training_validation.log
if ( $validation_status != 0 ) then
    echo "ERROR: training-data validation failed"
    exit $validation_status
endif

sha256sum $events > eirene_training_v3.sha256

echo "restart_source=${restart_source}" > eirene_training_provenance.txt
echo "generated_b2fpardf=${generated_b2fpardf}" >> eirene_training_provenance.txt
echo "generated_b2frates=${generated_b2frates}" >> eirene_training_provenance.txt
git -C "$solps_root" rev-parse HEAD >> eirene_training_provenance.txt
git -C "${solps_root}/modules/B2.5" rev-parse HEAD >> eirene_training_provenance.txt
touch EIRENE_TRAINING_SUCCESS

echo "PASS: $case_dir"
