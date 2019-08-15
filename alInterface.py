from enum import Enum
import sqlite3
import argparse
import collections
import slurmInterface
import os

class FineGrainProvider(Enum):
    LAMMPS = 0
    MYSTIC = 1
    ACTIVELEARNER=2
    FAKE = 3

ICFInputs = collections.namedtuple('ICFInputs', 'Temperature Density Charges')
ICFOutputs = collections.namedtuple('ICFOutputs', 'Viscosity ThermalConductivity DiffCoeff')

def buildAndLaunchLAMMPSJob(rank, tag, dbPath, uname, lammps, reqid, icfArgs):
    # Mkdir ./${TAG}_${RANK}_${REQ}
    outDir = tag + "_" + str(rank) + "_" + str(req)
    outPath = os.path.join(os.getcwd(), outDir)
    os.mkdir(outPath)
    # cp /lammpsScripts/in.Argon_Deuterium_plasma # Need to verify we can handle paths properly
    # generate slurm script by writing to file
    # either syscall or subprocess.run slurm with the script
    # Then do nothing because the script itself will write the result
    pass

def pollAndProcessFGSRequests(rankArr, mode, dbPath, tag, lammps, uname, maxJobs):
    reqNumArr = [0] * len(rankArr)

    # TODO: Figure out a way to stop that isn't ```kill - 9```
    while True:
        for i in range(0, len(rankArr)):
            rank = rankArr[i]
            req = reqNumArr[i]
            sqlDB = sqlite3.connect(dbPath)
            sqlCursor = sqlDB.cursor()
            selString = "SELECT * FROM REQS WHERE RANK=? AND REQ=? AND TAG=?;"
            selArgs = (rank, req, tag)
            reqQueue = []
            # Get current set of requests
            for row in sqlCursor.execute(selString, selArgs):
                #Process row to arguments
                icfInput = ICFInputs(Temperature=float(row[3]), Density=[], Charges=[])
                icfInput.Density.extend([row[4], row[5], row[6], row[7]])
                icfInput.Charges.extend([row[8], row[9], row[10], row[11]])
                #Enqueue task
                reqQueue.append((req, icfInput))
                #Increment reqNum
                reqNumArr[i] = reqNumArr[i] + 1
            sqlCursor.close()
            sqlDB.close()
            # Process tasks based on mode
            for task in reqQueue:
                if mode == FineGrainProvider.LAMMPS:
                    # call lammps with args as slurmjob
                    # slurmjob will write result back
                    launchedJob = False
                    while(launchedJob == False):
                        if slurmInterface.getSlurmQueue[0] < maxJobs:
                            buildAndLaunchLAMMPSJob(rank, tag, dbPath, uname, lammps, task[0], task[1])
                            launchedJob = True
                elif mode == FineGrainProvider.MYSTIC:
                    # call mystic: I think Mystic will handle most of our logic?
                    # TODO
                    pass
                elif mode == FineGrainProvider.ACTIVELEARNER:
                    # This is probably more for the Nick stuff
                    #  Ask Learner
                    #     We good? Return value
                    #  Check LUT
                    #     We good? Return value
                    #  Call LAMMPS
                    #     Go get a coffee, then return value. And add to LUT (?)
                    # TODO
                    pass
                elif mode == FineGrainProvider.FAKE:
                    # Simplest stencil imaginable
                    icfInput = task[1]
                    icfOutput = ICFOutputs(Viscosity=0.0, ThermalConductivity=0.0, DiffCoeff=[0.0]*10)
                    icfOutput.DiffCoeff[7] = (icfInput.Temperature + icfInput.Density[0] +  icfInput.Charges[3]) / 3
                    # Write the result
                    sqlDB = sqlite3.connect(dbPath)
                    sqlCursor = sqlDB.cursor()
                    insString = "INSERT INTO RESULTS VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    insArgs = (tag, rank, task[0], icfOutput.Viscosity, icfOutput.ThermalConductivity) + tuple(icfOutput.DiffCoeff)
                    sqlCursor.execute(insString, insArgs)
                    sqlDB.commit()
                    sqlCursor.close()
                    sqlDB.close()
        #Probably some form of delay?


if __name__ == "__main__":
    defaultFName = "testDB.db"
    defaultTag = "DUMMY_TAG_42"
    defaultLammps = "./lmp"
    defaultUname = "tcg"
    defaultMaxJobs = 4

    argParser = argparse.ArgumentParser(description='Python Shim for LAMMPS and AL')

    argParser.add_argument('-t', '--tag', action='store', type=str, required=False, default=defaultTag, help="Tag for DB Entries")
    argParser.add_argument('-l', '--lammps', action='store', type=str, required=False, default=defaultLammps, help="Path to LAMMPS Binary")
    argParser.add_argument('-d', '--db', action='store', type=str, required=False, default=defaultFName, help="Filename for sqlite DB")
    argParser.add_argument('-u', '--uname', action='store', type=str, required=False, default=defaultUname, help="Username to Query Slurm With")
    argParser.add_argument('-j', '--maxjobs', action='store', type=int, required=False, default=defaultMaxJobs, help="Maximum Number of Slurm Jobs To Enqueue")


    args = vars(argParser.parse_args())

    tag = args['tag']
    fName = args['db']
    lammps = args['lammps']
    uname = args['uname']
    jobs = args['maxjobs']

    pollAndProcessFGSRequests([0], FineGrainProvider.FAKE, fName, tag, lammps, uname, jobs)
