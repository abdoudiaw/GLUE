from enum import Enum, IntEnum
import sqlite3
import argparse
import collections
from zbar import zBar
from SM import Wigner_Seitz_radius
import os
import shutil
import numpy as np
import csv
import time
import subprocess

class ALInterfaceMode(Enum):
    LAMMPS = 0
    MYSTIC = 1
    ACTIVELEARNER=2
    FAKE = 3
    DEFAULT = 4
    KILL = 9

class SolverCode(Enum):
    BGK = 0
    LBMZEROD = 1

class ResultProvenance(IntEnum):
    FAKE = 0
    LAMMPS = 1
    INTERPOLATION = 2

BGKInputs = collections.namedtuple('BGKInputs', 'Temperature Density Charges')
BGKOutputs = collections.namedtuple('BGKOutputs', 'Viscosity ThermalConductivity DiffCoeff')

def getSelString(packetType):
    if packetType == SolverCode.BGK:
        return "SELECT * FROM BGKREQS WHERE RANK=? AND REQ=? AND TAG=?;"
    else:
        raise Exception('Using Unsupported Solver Code')

def processReqRow(sqlRow, packetType):
    if packetType == SolverCode.BGK:
        #Process row to arguments
        bgkInput = BGKInputs(Temperature=float(sqlRow[3]), Density=[], Charges=[])
        bgkInput.Density.extend([sqlRow[4], sqlRow[5], sqlRow[6], sqlRow[7]])
        bgkInput.Charges.extend([sqlRow[8], sqlRow[9], sqlRow[10], sqlRow[11]])
        reqType = ALInterfaceMode(sqlRow[12])
        return (bgkInput, reqType)
    else:
        raise Exception('Using Unsupported Solver Code')

def writeLammpsInputs(lammpsArgs, dirPath):
    # TODO: Refactor constants and general cleanup
    # WARNING: Seems to be restricted to two materials for now
    if isinstance(lammpsArgs, BGKInputs):
        m=np.array([3.3210778e-24,6.633365399999999e-23])
        Z=np.array([1,13])
        interparticle_radius = []
        lammpsDens = np.array(lammpsArgs.Density[0:2])
        lammpsTemperature = lammpsArgs.Temperature
        lammpsIonization = np.array(lammpsArgs.Charges[0:2])
        for s in range(len(lammpsDens)):
            zbarFile = os.path.join(dirPath, "Zbar." + str(s) + ".csv")
            with open(zbarFile, 'w') as testfile:
                csv_writer = csv.writer(testfile,delimiter=' ')
                csv_writer.writerow([lammpsIonization[s-1]])
        temperatureFile = os.path.join(dirPath, "temperature.csv")
        with open(temperatureFile, 'w') as testfile:
            csv_writer = csv.writer(testfile,delimiter=' ')
            csv_writer.writerow([lammpsTemperature])
        interparticle_radius.append(Wigner_Seitz_radius(sum(lammpsDens)))
        L=20*max(interparticle_radius)  #in cm
        volume =L**3
        boxLengthFile = os.path.join(dirPath, "box_length.csv")
        with open(boxLengthFile, 'w') as testfile:
            csv_writer = csv.writer(testfile,delimiter=' ')
            csv_writer.writerow([L*1.e-2])
        N=[]
        for s in range(len(lammpsDens)):
            N.append(int(volume*lammpsDens[s]))
            numberPartFile = zbarFile = os.path.join(dirPath, "Number_part." + str(s) + ".csv")
            with open(numberPartFile, 'w') as testfile:
                csv_writer = csv.writer(testfile,delimiter=' ')
                csv_writer.writerow([N[s]])
    else:
        raise Exception('Using Unsupported Solver Code')

def checkSlurmQueue(uname):
    try:
        runproc = subprocess.run(
            ["squeue", "-u", uname],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        if runproc.returncode == 0:
            return str(runproc.stdout,"utf-8")
        else:
            print(str(runproc.stderr,"utf-8"), file=sys.stderr)
            return ""
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return ""

def launchSlurmJob(script):
    try:
        runproc = subprocess.run(
            ["sbatch", script],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        if runproc.returncode == 0:
            return str(runproc.stdout,"utf-8")
        else:
            print(str(runproc.stderr,"utf-8"), file=sys.stderr)
            return ""
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return ""

def getSlurmQueue(uname):
    slurmOut = checkSlurmQueue(uname)
    if slurmOut == "":
        return (sys.maxsize, [])
    strList = slurmOut.splitlines()
    if len(strList) > 1:
        return (len(strList) - 1, strList[1:])
    else:
        return (0, [])

def buildAndLaunchLAMMPSJob(rank, tag, dbPath, uname, lammps, reqid, lammpsArgs):
    if isinstance(lammpsArgs, BGKInputs):
        # Mkdir ./${TAG}_${RANK}_${REQ}
        outDir = tag + "_" + str(rank) + "_" + str(reqid)
        outPath = os.path.join(os.getcwd(), outDir)
        if(not os.path.exists(outPath)):
            os.mkdir(outPath)
            # cp ${SCRIPT_DIR}/lammpsScripts/in.Argon_Deuterium_plasma 
            # Copy scripts and configuration files
            pythonScriptDir = os.path.dirname(os.path.realpath(__file__))
            lammpsPath = os.path.join(pythonScriptDir, "lammpsScripts")
            ardPlasPath = os.path.join(lammpsPath, "in.Argon_Deuterium_plasma")
            slurmEnvPath = os.path.join(pythonScriptDir, "slurmScripts")
            jobEnvFilePath = os.path.join(slurmEnvPath, "jobEnv.sh")
            shutil.copy2(ardPlasPath, outPath)
            shutil.copy2(jobEnvFilePath, outPath)
            bgkResultScript = os.path.join(pythonScriptDir, "processBGKResult.py")
            # Generate input files
            writeLammpsInputs(lammpsArgs, outPath)
            # Generate slurm script by writing to file
            # TODO: Identify a cleaner way to handle QOS and accounts and all the fun slurm stuff?
            # TODO: DRY this
            slurmFPath = os.path.join(outPath, tag + "_" + str(rank) + "_" + str(reqid) + ".sh")
            with open(slurmFPath, 'w') as slurmFile:
                slurmFile.write("#!/bin/bash\n")
                slurmFile.write("#SBATCH -N 1\n")
                slurmFile.write("#SBATCH -o " + tag + "-%j.out\n")
                slurmFile.write("#SBATCH -e " + tag + "-%j.err\n")
                slurmFile.write("cd " + outPath + "\n")
                slurmFile.write("source ./jobEnv.sh\n")
                # Actually call lammps
                slurmFile.write("srun -n 4 " + lammps + " < in.Argon_Deuterium_plasma   \n")
                # Process the result and write to DB
                slurmFile.write("python3 " + bgkResultScript + " -t " + tag + " -r " + str(rank) + " -i " + str(reqid) + " -d " + os.path.realpath(dbPath) + " -f ./mutual_diffusion.csv\n")
            # either syscall or subprocess.run slurm with the script
            launchSlurmJob(slurmFPath)
            # Then do nothing because the script itself will write the result
    else:
        raise Exception('Using Unsupported Solver Code')

def insertResult(rank, tag, dbPath, reqid, lammpsResult, resultProvenance):
    if isinstance(lammpsResult, BGKOutputs):
        sqlDB = sqlite3.connect(dbPath)
        sqlCursor = sqlDB.cursor()
        insString = "INSERT INTO BGKRESULTS VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        insArgs = (tag, rank, reqid, lammpsResult.Viscosity, lammpsResult.ThermalConductivity) + tuple(lammpsResult.DiffCoeff) + (resultProvenance,)
        sqlCursor.execute(insString, insArgs)
        sqlDB.commit()
        sqlCursor.close()
        sqlDB.close()
    else:
        raise Exception('Using Unsupported Solver Code')

def pollAndProcessFGSRequests(rankArr, defaultMode, dbPath, tag, lammps, uname, maxJobs, sbatch, packetType):
    reqNumArr = [0] * len(rankArr)

    #Spin until file exists
    while not os.path.exists(dbPath):
        time.sleep(1)

    keepSpinning = True
    while keepSpinning:
        for i in range(0, len(rankArr)):
            rank = rankArr[i]
            req = reqNumArr[i]
            sqlDB = sqlite3.connect(dbPath)
            # SELECT request
            sqlCursor = sqlDB.cursor()
            selString = getSelString(packetType)
            selArgs = (rank, req, tag)
            reqQueue = []
            # Get current set of requests
            for row in sqlCursor.execute(selString, selArgs):
                #Process row to arguments
                solverInput = processReqRow(row, packetType)
                #Enqueue task
                reqQueue.append((req,) + solverInput)
                #Increment reqNum
                reqNumArr[i] = reqNumArr[i] + 1
            sqlCursor.close()
            sqlDB.close()
            # Process tasks based on mode
            for task in reqQueue:
                modeSwitch = defaultMode
                if task[2] != ALInterfaceMode.DEFAULT:
                    modeSwitch = task[2]
                if modeSwitch == ALInterfaceMode.LAMMPS:
                    # This is a brute force call. We only want an exact LAMMPS result
                    # So first, check if we have already processed this request
                    #TODO
                    # Then, call lammps with args as slurmjob
                    # slurmjob will write result back
                    launchedJob = False
                    while(launchedJob == False):
                        queueState = getSlurmQueue(uname)
                        if queueState[0] < maxJobs:
                            print("Processing REQ=" + str(task[0]))
                            buildAndLaunchLAMMPSJob(rank, tag, dbPath, uname, lammps, task[0], task[1])
                            launchedJob = True
                elif modeSwitch == ALInterfaceMode.MYSTIC:
                    # call mystic: I think Mystic will handle most of our logic?
                    # TODO
                    pass
                elif modeSwitch == ALInterfaceMode.ACTIVELEARNER:
                    # This is probably more for the Nick stuff
                    #  Ask Learner
                    #     We good? Return value
                    #  Check LUT
                    #     We good? Return value
                    #  Call LAMMPS
                    #     Go get a coffee, then return value. And add to LUT (?)
                    # TODO
                    pass
                elif modeSwitch == ALInterfaceMode.FAKE:
                    if packetType == SolverCode.BGK:
                        # Simplest stencil imaginable
                        bgkInput = task[1]
                        bgkOutput = BGKOutputs(Viscosity=0.0, ThermalConductivity=0.0, DiffCoeff=[0.0]*10)
                        bgkOutput.DiffCoeff[7] = (bgkInput.Temperature + bgkInput.Density[0] +  bgkInput.Charges[3]) / 3
                        # Write the result
                        insertResult(rank, tag, dbPath, task[0], bgkOutput, ResultProvenance.FAKE)
                    else:
                        raise Exception('Using Unsupported Solver Code')
                elif modeSwitch == ALInterfaceMode.KILL:
                    keepSpinning = False
        #Probably some form of delay?


if __name__ == "__main__":
    defaultFName = "testDB.db"
    defaultTag = "DUMMY_TAG_42"
    defaultLammps = "./lmp"
    defaultUname = "tcg"
    defaultSqlite = "sqlite3"
    defaultSbatch = "/usr/bin/sbatch"
    defaultMaxJobs = 4
    defaultProcessing = ALInterfaceMode.LAMMPS
    defaultRanks = [0]
    defaultSolver = SolverCode.BGK

    argParser = argparse.ArgumentParser(description='Python Shim for LAMMPS and AL')

    argParser.add_argument('-t', '--tag', action='store', type=str, required=False, default=defaultTag, help="Tag for DB Entries")
    argParser.add_argument('-l', '--lammps', action='store', type=str, required=False, default=defaultLammps, help="Path to LAMMPS Binary")
    argParser.add_argument('-q', '--sqlite', action='store', type=str, required=False, default=defaultSqlite, help="Path to sqlite3 Binary")
    argParser.add_argument('-s', '--sbatch', action='store', type=str, required=False, default=defaultSbatch, help="Path to sbatch Binary")
    argParser.add_argument('-d', '--db', action='store', type=str, required=False, default=defaultFName, help="Filename for sqlite DB")
    argParser.add_argument('-u', '--uname', action='store', type=str, required=False, default=defaultUname, help="Username to Query Slurm With")
    argParser.add_argument('-j', '--maxjobs', action='store', type=int, required=False, default=defaultMaxJobs, help="Maximum Number of Slurm Jobs To Enqueue")
    argParser.add_argument('-m', '--mode', action='store', type=int, required=False, default=defaultProcessing, help="Default Request Type (LAMMPS=0)")
    argParser.add_argument('-r', '--ranks', nargs='+', default=defaultRanks, type=int,  help="Rank IDs to Listen For")
    argParser.add_argument('-c', '--code', action='store', type=int, required=False, default=defaultSolver, help="Code to expect Packets from (BGK=0)")


    args = vars(argParser.parse_args())

    tag = args['tag']
    fName = args['db']
    lammps = args['lammps']
    uname = args['uname']
    jobs = args['maxjobs']
    sqlite = args['sqlite']
    sbatch = args['sbatch']
    ranks = args['ranks']
    mode = ALInterfaceMode(args['mode'])
    code = SolverCode(args['code'])

    pollAndProcessFGSRequests(ranks, mode, fName, tag, lammps, uname, jobs, sbatch, code)
