#ifndef RABITQ_H
#define RABITQ_H

#include "postgres.h"
#include "fmgr.h"
#include "vector.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#define RABITQ_VERSION 1
#define RABITQ_MAX_DIM 4000
#define RABITQ_BYTES(dim, bits) (((dim) * (bits) + 7) / 8)
#define RABITQ_SIZE(dim, bits) (offsetof(Rabitq, code) + RABITQ_BYTES(dim, bits))
#define PG_GETARG_RABITQ_P(n) ((Rabitq *) PG_DETOAST_DATUM(PG_GETARG_DATUM(n)))

/* The version covers the rotation, packing and estimator, not just I/O */
typedef struct Rabitq
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int16		dim;
	uint8		bits;
	uint8		version;
	double		norm2;
	double		scale;
	uint8		code[FLEXIBLE_ARRAY_MEMBER];
}			Rabitq;

typedef struct RabitqQuery
{
	int			dim;
	double		norm2;
	double	   *rotated;
	double	   *lut;
}			RabitqQuery;

Rabitq	   *RabitqEncode(const float *values, int dimensions, int bits);
RabitqQuery *RabitqPrepareQuery(const float *values, int dimensions);
double		RabitqQueryIP(const Rabitq *value, const RabitqQuery *query);
double		RabitqCodeIP(const Rabitq *a, const Rabitq *b);

#endif
