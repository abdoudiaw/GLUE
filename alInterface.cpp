#include "alInterface.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iterator>
#include <algorithm>
#include <sqlite3.h>
#include <mpi.h>

#ifdef DB_EXISTENCE_SPIN
#include <experimental/filesystem>
#include <thread>
#include <chrono>
#endif

///TODO: Verify this is the correct way to do a global variable
AsyncSelectTable_t<bgk_result_t> globalBGKResultTable;
AsyncSelectTable_t<lbmToOneDMD_result_t> globallbmToOneDMDResultTable;
sqlite3* globalGlueDBHandle;
const unsigned int globalGlueBufferSize = 1024;

static int dummyCallback(void *NotUsed, int argc, char **argv, char **azColName)
{
	//Do nothing. We don't need a result from this op
	return 0;
}

static int readCallback_bgk(void *NotUsed, int argc, char **argv, char **azColName)
{
	//Process row: Ignore 0 (tag) and 1 (rank)
	int reqID = atoi(argv[2]);
	bgk_result_t result;

	//Add results
	result.viscosity = atof(argv[3]);
	result.thermalConductivity = atof(argv[4]);
	for(int i = 0; i < 10; i++)
	{
		result.diffusionCoefficient[i] = atof(argv[i+5]);
	}
	result.provenance = atoi(argv[15]);

	//Get global select table of type bgk_result_t
	globalBGKResultTable.tableMutex.lock();
	//Check if request has been processed yet
	auto reqIter = globalBGKResultTable.resultTable.find(reqID);
	if (reqIter == globalBGKResultTable.resultTable.end())
	{
		//Write result to global map so we can use it
		globalBGKResultTable.resultTable[reqID] = result;
	}
	globalBGKResultTable.tableMutex.unlock();

	return 0;
}

int getReqNumber()
{
	//Static variables are dirty but this is an okay use
	static int reqNumber = 0;
	int retNum = reqNumber;
	reqNumber++;
	return retNum;
}

bgk_result_t bgk_req_single_with_reqtype(bgk_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType)
{
	return req_single_with_reqtype<bgk_request_t, bgk_result_t>(input, mpiRank, tag, dbHandle, reqType);
}

bgk_result_t bgk_req_single(bgk_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle)
{
	return bgk_req_single_with_reqtype(input, mpiRank, tag, dbHandle, ALInterfaceMode_e::DEFAULT);
}

bgk_result_t* bgk_req_batch_with_reqtype(bgk_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType)
{
	return req_batch_with_reqtype<bgk_request_t, bgk_result_t>(input, numInputs, mpiRank, tag, dbHandle, reqType);
}

bgk_result_t* bgk_req_batch(bgk_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle)
{
	return bgk_req_batch_with_reqtype(input, numInputs, mpiRank, tag, dbHandle, ALInterfaceMode_e::DEFAULT);
}

void bgk_stop_service(int mpiRank, char * tag, sqlite3 *dbHandle)
{
	bgk_request_t req;
	req.temperature = -0.0;
	for(int i = 0; i < 4; i++)
	{
		req.density[i] = -0.0;
		req.charges[i] = -0.0;
	}

	bgk_req_single_with_reqtype(req, mpiRank, tag, dbHandle, ALInterfaceMode_e::KILL);
	return;
}

bgkmasses_result_t bgkmasses_req_single_with_reqtype(bgkmasses_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType)
{
	return req_single_with_reqtype<bgkmasses_request_t, bgkmasses_result_t>(input, mpiRank, tag, dbHandle, reqType);
}

bgkmasses_result_t bgkmasses_req_single(bgkmasses_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle)
{
	return bgkmasses_req_single_with_reqtype(input, mpiRank, tag, dbHandle, ALInterfaceMode_e::DEFAULT);
}

bgkmasses_result_t* bgkmasses_req_batch_with_reqtype(bgkmasses_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType)
{
	return req_batch_with_reqtype<bgkmasses_request_t, bgkmasses_result_t>(input, numInputs, mpiRank, tag, dbHandle, reqType);
}

bgkmasses_result_t* bgkmasses_req_batch(bgkmasses_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle)
{
	return bgkmasses_req_batch_with_reqtype(input, numInputs, mpiRank, tag, dbHandle, ALInterfaceMode_e::DEFAULT);
}

void bgkmasses_stop_service(int mpiRank, char * tag, sqlite3 *dbHandle)
{
	bgkmasses_request_t req;
	req.temperature = -0.0;
	for(int i = 0; i < 4; i++)
	{
		req.density[i] = -0.0;
		req.charges[i] = -0.0;
		req.masses[i] = -0.0;
	}

	bgkmasses_req_single_with_reqtype(req, mpiRank, tag, dbHandle, ALInterfaceMode_e::KILL);
	return;
}

lbmToOneDMD_result_t lbmToOneDMD_req_single(lbmToOneDMD_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle)
{
	return lbmToOneDMD_req_single_with_reqtype(input, mpiRank, tag, dbHandle, ALInterfaceMode_e::DEFAULT);
}

lbmToOneDMD_result_t lbmToOneDMD_req_single_with_reqtype(lbmToOneDMD_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType)
{
	return req_single_with_reqtype<lbmToOneDMD_request_t, lbmToOneDMD_result_t>(input, mpiRank, tag, dbHandle, reqType);
}

lbmToOneDMD_result_t* lbmToOneDMD_req_batch_with_reqtype(lbmToOneDMD_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType)
{
	return req_batch_with_reqtype<lbmToOneDMD_request_t, lbmToOneDMD_result_t>(input, numInputs, mpiRank, tag, dbHandle, reqType);
}

lbmToOneDMD_result_t* lbmToOneDMD_req_batch(lbmToOneDMD_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle)
{
	return lbmToOneDMD_req_batch_with_reqtype(input, numInputs, mpiRank, tag, dbHandle, ALInterfaceMode_e::DEFAULT);
}

void lbmToOneDMD_stop_service(int mpiRank, char * tag, sqlite3 *dbHandle)
{
	lbmToOneDMD_request_t req;
	req.distance = -0.0;
	req.density = -0.0;
	req.temperature = -0.0;

	lbmToOneDMD_req_single_with_reqtype(req, mpiRank, tag, dbHandle, ALInterfaceMode_e::KILL);
	return;
}

sqlite3* initDB(int mpiRank, char * fName)
{
#ifdef DB_EXISTENCE_SPIN
	while(!std::experimental::filesystem::exists(fName))
	{
		std::this_thread::sleep_for (std::chrono::seconds(1));
	}
#endif
	sqlite3 *dbHandle;
	sqlite3_open(fName, &dbHandle);
	return dbHandle;
}

void closeDB(sqlite3* dbHandle)
{
	sqlite3_close(dbHandle);
}

void resFreeWrapper(void * buffer)
{
	free(buffer);
}

void connectGlue(char * fName, MPI_Comm glueComm)
{
	//Not a collective operation because only rank 0 needs to do this
	//If glueComm rank is 0
	int myRank;
	MPI_Comm_rank(glueComm, &myRank);
	if(myRank == 0)
	{
		//Connect to that DB
		sqlite3_open(fName, &globalGlueDBHandle);
	}
}

void preprocess_icf(bgk_request_t *input, int numInputs, bgk_request_t **processedInput, int * numProcessedInputs)
{
	///TODO
	//Look for and remove duplicates
	return;
}

bgk_result_t* icf_req(bgk_request_t *input, int numInputs, MPI_Comm glueComm)
{
	///TODO: Will probably refactor to a template
	//Get clueComm rank and size
	int myRank, commSize;
	MPI_Comm_rank(glueComm, &myRank);
	MPI_Comm_size(glueComm, &commSize);
	//Compute number of required request batches
	std::vector<int> batchBuffer(commSize, 0);
	batchBuffer[myRank] = numInputs / globalGlueBufferSize;
	if(numInputs % globalGlueBufferSize != 0)
	{
		batchBuffer[myRank]++;
	}
	//Reduce to provide that to 0
	MPI_Reduce(batchBuffer.data(), batchBuffer.data(), commSize, MPI_INT, MPI_MAX, 0, glueComm);
	//Prepare results buffer
	bgk_result_t* reqBuffer = (bgk_result_t*)malloc(sizeof(bgk_result_t*) * numInputs);
	//If rank 0
	if(myRank == 0)
	{
		//First, submit all rank 0 requests
		///TODO
		//Then, do the rest
		std::vector<int> resultBatches(batchBuffer);
		///TODO: Need to preserve range of results we expect and number of results
		for(int rank = 1; rank < commSize; rank++)
		{
			//Do we still have requests from that rank?
			if(batchBuffer[rank] != 0)
			{
				//recv those requests
				///TODO
				//Send to glue code
				///TODO
				batchBuffer[rank]--;
			}
		}
		//Now, we process results
		for(int rank = 1; rank < commSize; rank++)
		{
			//Do we still have results for that rank?
			if(resultBatches[rank] != 0)
			{
				///TODO
				resultBatches[rank]--;
			}
		}
		//And then handle the requests from rank 0
		///TODO
	}
	else
	{
		//Everyone else, send your requests to rank 0
		for(int i = 0; i < batchBuffer[myRank]; i++)
		{
			//Send requests to buffer on rank 0
			///TODO
			//Wait for results
			///TODO
		}
	}
	//Return results
	return reqBuffer;
}

void closeGlue(MPI_Comm glueComm)
{
	//Not a collective operation because only rank 0 needs to do this
	//If glueComm rank is 0
	int myRank;
	MPI_Comm_rank(glueComm, &myRank);
	if(myRank == 0)
	{
		///TODO: Send killswitch
		//Disconnect from that DB
		sqlite3_close(globalGlueDBHandle);
	}
}
