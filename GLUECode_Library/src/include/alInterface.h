#ifndef __alInterface_h
#define __alInterface_h

#include <math.h>
#include <mpi.h>
#ifdef SOLVER_SIDE_SQLITE
#include <sqlite3.h>
#endif

enum ALInterfaceMode_e
{
	FGS = 0,
	MYSTIC = 1,
	ACTIVELEARNER = 2,
	FAKE = 3,
	DEFAULT = 4,
	FASTFGS = 5,
	ANALYTIC = 6,
	KILL = 9
};

enum DatabaseMode_e
{
	SQLITE = 0,
	MYSQL = 1,
	HDF5 = 2
};

/**
 * @brief Struct to represent ICF fine grain simulation request for the BGK application
 */
struct bgk_request_s
{
	double temperature;
	//n
	double density[4];
	double charges[4];

	#ifdef __cplusplus
	
	/**
	 * @brief This is a C++ function that is used to compare two bgk_request_s structs. It is used to determine if a request has already been made.
	 * 
	 * @param lhs First request
	 * @param rhs Second request
	 * @return true Requests are equal within given tolerance
	 * @return false Requests are not equal within given tolerance
	 */
	friend bool operator==(const bgk_request_s& lhs, const bgk_request_s& rhs)
	{
		bool areEqual = true;
		const double tempEpsilon = 0.0001;
		const double densEpsilon = 0.0001;
		const double chargeEpsilon = 0.0001;
		if (fabs(lhs.temperature - rhs.temperature)/ rhs.temperature > tempEpsilon) areEqual = false;
		for(int i = 0; i < 4; i++)
		{
			if (fabs(lhs.density[i] - rhs.density[i]) /rhs.density[i] > densEpsilon) areEqual = false;
			if (fabs(lhs.charges[i] - rhs.charges[i]) / rhs.charges[i] > chargeEpsilon) areEqual = false;
		}
		return areEqual;
	}
	#endif
};

/**
 * @brief Struct to represent ICF fine grain simulation result for the BGK application
 */
struct bgk_result_s
{
	double viscosity;
	double thermalConductivity;
	//n*n+1/2
	double diffusionCoefficient[10];
	int provenance;

	#ifdef __cplusplus
    /**
	 * @brief This is a C++ function that is used to compare two bgk_result_s structs. It is used to determine if a result is already available.
	 * 
	 * @param lhs First result
	 * @param rhs Second result
	 * @return true Results are equal within given tolerance
	 * @return false Results are not equal within given tolerance
	 */
	friend bool operator==(const bgk_result_s& lhs, const bgk_result_s& rhs)
	{
		bool areEqual = true;
		const double viscEpsilon = 0.0001;
		const double thermCondEpsilon = 0.0001;
		const double diffEpsilon = 0.0001;
		if (fabs(lhs.viscosity - rhs.viscosity) / rhs.viscosity > viscEpsilon) areEqual = false;
		if (fabs(lhs.thermalConductivity - rhs.thermalConductivity) / rhs.thermalConductivity > thermCondEpsilon) areEqual = false;
		for(int i = 0; i < 10; i++)
		{
			if (fabs(lhs.diffusionCoefficient[i] - rhs.diffusionCoefficient[i]) / rhs.diffusionCoefficient[i] > diffEpsilon) areEqual = false;
		}
		return areEqual;
	}
	#endif
};

/**
 * @brief Struct to represent ICF fine grain simulation request for the BGK application where species are identified by mass
 */
struct bgkmasses_request_s
{
	double temperature;
	//n
	double density[4];
	double charges[4];
	double masses[4];
};

/**
 * @brief Struct to represent ICF fine grain simulation result for the BGK application where species are identified by mass
 */
struct bgkmasses_result_s
{
	double viscosity;
	double thermalConductivity;
	//n*n+1/2
	double diffusionCoefficient[10];
};

/**
 * @brief Struct to represent Shale simulation requests for the LBM application in a 1-Dimensional Problem
 */
struct lbmToOneDMD_request_s
{
	double distance;
	double density;
	double temperature;
};

/**
 * @brief Struct to represent Shale simulation results for the LBM application in a 1-Dimensional Problem
 */
struct lbmToOneDMD_result_s
{
	double adsorption;
	int provenance;
};

typedef struct bgk_result_s bgk_result_t;
typedef struct bgk_request_s bgk_request_t;
typedef struct bgkmasses_result_s bgkmasses_result_t;
typedef struct bgkmasses_request_s bgkmasses_request_t;
typedef struct lbmToOneDMD_result_s lbmToOneDMD_result_t;
typedef struct lbmToOneDMD_request_s lbmToOneDMD_request_t;

#ifdef SOLVER_SIDE_SQLITE
typedef sqlite3 * dbHandle_t;
#else
typedef void * dbHandle_t;
#endif

#ifdef __cplusplus
extern "C"
{
#endif
	bgk_result_t bgk_req_single(bgk_request_t input, int mpiRank, char * tag, dbHandle_t dbHandle);
	bgk_result_t bgk_req_single_with_reqtype(bgk_request_t input, int mpiRank, char * tag, dbHandle_t dbHandle, unsigned int reqType);
	bgk_result_t* bgk_req_batch(bgk_request_t *input, int numInputs, int mpiRank, char * tag, dbHandle_t dbHandle);
	bgk_result_t* bgk_req_batch_with_reqtype(bgk_request_t *input, int numInputs, int mpiRank, char * tag, dbHandle_t dbHandle, unsigned int reqType);
	void bgk_stop_service(int mpiRank, char * tag, dbHandle_t dbHandle);

	bgkmasses_result_t bgkmasses_req_single(bgkmasses_request_t input, int mpiRank, char * tag, dbHandle_t dbHandle);
	bgkmasses_result_t bgkmasses_req_single_with_reqtype(bgkmasses_request_t input, int mpiRank, char * tag, dbHandle_t dbHandle, unsigned int reqType);
	bgkmasses_result_t* bgkmasses_req_batch(bgkmasses_request_t *input, int numInputs, int mpiRank, char * tag, dbHandle_t dbHandle);
	bgkmasses_result_t* bgkmasses_req_batch_with_reqtype(bgkmasses_request_t *input, int numInputs, int mpiRank, char * tag, dbHandle_t dbHandle, unsigned int reqType);
	void bgkmasses_stop_service(int mpiRank, char * tag, dbHandle_t dbHandle);

	lbmToOneDMD_result_t lbmToOneDMD_req_single(lbmToOneDMD_request_t input, int mpiRank, char * tag, dbHandle_t  dbHandle);
	lbmToOneDMD_result_t lbmToOneDMD_req_single_with_reqtype(lbmToOneDMD_request_t input, int mpiRank, char * tag, dbHandle_t dbHandle, unsigned int reqType);
	lbmToOneDMD_result_t* lbmToOneDMD_req_batch(lbmToOneDMD_request_t *input, int numInputs, int mpiRank, char * tag, dbHandle_t dbHandle);
	lbmToOneDMD_result_t* lbmToOneDMD_req_batch_with_reqtype(lbmToOneDMD_request_t *input, int numInputs, int mpiRank, char * tag, dbHandle_t dbHandle, unsigned int reqType);
	void lbmToOneDMD_stop_service(int mpiRank, char * tag, dbHandle_t dbHandle);

	void resFreeWrapper(void * buffer);
	
	dbHandle_t initDB(int mpiRank, char * fName);
	void closeDB(dbHandle_t dbHandle);

	/**
	 * @brief Interface to connect to GLUE Code service for MPI Collective based approach
	 * 
	 * @param fName String used to connect to SQL Database
	 * @param glueComm MPI Communicator to be used by GLUE Code
	 */
	void connectGlue(char * fName, MPI_Comm glueComm);
	/**
	 * @brief Preprocess ICF Requests prior to sending to GLUE Code
	 * 
	 * @param input Array of inputs to request fine grain simulations for ICF applications using BGK format
	 * @param numInputs Length of input
	 * @param processedInput Pointer to preprocessed array
	 * @param numProcessedInputs Length of processedInput
	 */
	void preprocess_icf(bgk_request_t *input, int numInputs, bgk_request_t **processedInput, int * numProcessedInputs);
	/**
	 * @brief Batch request of fine grain simulations for ICF applications using BGK format as an MPI collective operation
	 * 
	 * @param input Array of inputs to request fine grain simulations for ICF applications using BGK format
	 * @param numInputs Length of input
	 * @param glueComm MPI Communicator to be used by GLUE Code
	 * @return bgk_result_t* Array of results for fine grain simulations of length numInputs
	 */
	bgk_result_t* icf_req(bgk_request_t *input, int numInputs, MPI_Comm glueComm);
	/**
	 * @brief Close connection to GLUE Code service for MPI Collective based approach
	 * 
	 * @param glueComm MPI Communicator to be used by GLUE Code
	 */
	void closeGlue(MPI_Comm glueComm);

#ifdef __cplusplus
}
#endif

#endif /* __alInterface_h */
