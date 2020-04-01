#ifndef __alInterface_h
#define __alInterface_h

#include <sqlite3.h>
#include <math.h>

enum ALInterfaceMode_e
{
	LAMMPS = 0,
	MYSTIC = 1,
	ACTIVELEARNER = 2,
	FAKE = 3,
	DEFAULT = 4,
	FASTLAMMPS = 5,
	ANALYTIC = 6,
	KILL = 9
};

struct bgk_request_s
{
	double temperature;
	//n
	double density[4];
	double charges[4];

	#ifdef __cplusplus
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

struct bgk_result_s
{
	double viscosity;
	double thermalConductivity;
	//n*n+1/2
	double diffusionCoefficient[10];
	int provenance;

	#ifdef __cplusplus
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

struct bgkmasses_request_s
{
	double temperature;
	//n
	double density[4];
	double charges[4];
	double masses[4];
};

struct bgkmasses_result_s
{
	double viscosity;
	double thermalConductivity;
	//n*n+1/2
	double diffusionCoefficient[10];
};

struct lbmToOneDMD_request_s
{
	//compute distance, L, from the wall
	double distance;
	//density
	double density;
	//temperature L:
	double temperature;
};

struct lbmToOneDMD_result_s
{
	//adsorption
	double adsorption;
	int provenance;
};

typedef struct bgk_result_s bgk_result_t;
typedef struct bgk_request_s bgk_request_t;
typedef struct bgkmasses_result_s bgkmasses_result_t;
typedef struct bgkmasses_request_s bgkmasses_request_t;
typedef struct lbmToOneDMD_result_s lbmToOneDMD_result_t;
typedef struct lbmToOneDMD_request_s lbmToOneDMD_request_t;

#ifdef __cplusplus
extern "C"
{
#endif
	bgk_result_t bgk_req_single(bgk_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle);
	bgk_result_t bgk_req_single_with_reqtype(bgk_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType);
	bgk_result_t* bgk_req_batch(bgk_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle);
	bgk_result_t* bgk_req_batch_with_reqtype(bgk_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType);
	void bgk_stop_service(int mpiRank, char * tag, sqlite3 *dbHandle);

	bgkmasses_result_t bgkmasses_req_single(bgkmasses_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle);
	bgkmasses_result_t bgkmasses_req_single_with_reqtype(bgkmasses_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType);
	bgkmasses_result_t* bgkmasses_req_batch(bgkmasses_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle);
	bgkmasses_result_t* bgkmasses_req_batch_with_reqtype(bgkmasses_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType);
	void bgkmasses_stop_service(int mpiRank, char * tag, sqlite3 *dbHandle);

	lbmToOneDMD_result_t lbmToOneDMD_req_single(lbmToOneDMD_request_t input, int mpiRank, char * tag, sqlite3 * dbHandle);
	lbmToOneDMD_result_t lbmToOneDMD_req_single_with_reqtype(lbmToOneDMD_request_t input, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType);
	lbmToOneDMD_result_t* lbmToOneDMD_req_batch(lbmToOneDMD_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle);
	lbmToOneDMD_result_t* lbmToOneDMD_req_batch_with_reqtype(lbmToOneDMD_request_t *input, int numInputs, int mpiRank, char * tag, sqlite3 *dbHandle, unsigned int reqType);
	void lbmToOneDMD_stop_service(int mpiRank, char * tag, sqlite3 *dbHandle);

	void resFreeWrapper(void * buffer);
	
	sqlite3 * initDB(int mpiRank, char * fName);
	void closeDB(sqlite3* dbHandle);
#ifdef __cplusplus
}
#endif

#endif /* __alInterface_h */
