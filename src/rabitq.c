#include "postgres.h"

#include <float.h>
#include <math.h>

#include "hnsw.h"
#include "libpq/pqformat.h"
#include "rabitq.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/memutils.h"

static void
CheckDimensions(int dim, int typmod)
{
	if (dim < 1 || dim > RABITQ_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("rabitq must have between 1 and %d dimensions", RABITQ_MAX_DIM)));
	if (typmod >= 0 && typmod != dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("expected %d dimensions, not %d", typmod, dim)));
}

static void
CheckBits(int bits)
{
	if (bits != 1 && bits != 2 && bits != 4 && bits != 8)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("rabitq bits must be 1, 2, 4, or 8")));
}

static void
CheckVersion(const Rabitq *value)
{
	if (value->version != RABITQ_VERSION)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("unsupported rabitq format version %u", value->version)));
}

static void
CheckValue(Pointer pointer)
{
	Rabitq	   *value = (Rabitq *) pointer;
	int			used;

	CheckDimensions(value->dim, -1);
	CheckBits(value->bits);
	CheckVersion(value);
	if (VARSIZE(value) != RABITQ_SIZE(value->dim, value->bits))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid rabitq length")));
	if (!isfinite(value->norm2) || value->norm2 < 0 ||
		value->norm2 > (double) FLT_MAX * FLT_MAX * value->dim ||
		!isfinite(value->scale) || value->scale < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid rabitq factors")));
	/* Encoded scales never exceed the norm; the bound keeps products finite */
	if (value->scale > sqrt(value->norm2) * 1.000001)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("inconsistent rabitq factors")));
	used = (value->dim * value->bits) % 8;
	if (used && (value->code[RABITQ_BYTES(value->dim, value->bits) - 1] >> used))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("nonzero rabitq padding bits")));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_quantize);
Datum
rabitq_quantize(PG_FUNCTION_ARGS)
{
	Vector	   *value = PG_GETARG_VECTOR_P(0);
	int			bits = PG_GETARG_INT32(1);
	Rabitq	   *result;

	CheckDimensions(value->dim, -1);
	CheckBits(bits);
	result = RabitqEncode(value->x, value->dim, bits);
	PG_FREE_IF_COPY(value, 0);
	PG_RETURN_POINTER(result);
}

static Rabitq *
Receive(StringInfo buf, int typmod)
{
	int			version = pq_getmsgbyte(buf);
	int			bits = pq_getmsgbyte(buf);
	int			dim = pq_getmsgint(buf, 2);
	Rabitq	   *result;

	CheckDimensions(dim, typmod);
	CheckBits(bits);
	result = palloc0(RABITQ_SIZE(dim, bits));
	SET_VARSIZE(result, RABITQ_SIZE(dim, bits));
	result->version = version;
	result->bits = bits;
	result->dim = dim;
	result->norm2 = pq_getmsgfloat8(buf);
	result->scale = pq_getmsgfloat8(buf);
	pq_copymsgbytes(buf, (char *) result->code, RABITQ_BYTES(dim, bits));
	pq_getmsgend(buf);
	CheckValue((Pointer) result);
	return result;
}

static bytea *
Send(Rabitq *value)
{
	StringInfoData buf;

	pq_begintypsend(&buf);
	pq_sendbyte(&buf, value->version);
	pq_sendbyte(&buf, value->bits);
	pq_sendint16(&buf, value->dim);
	pq_sendfloat8(&buf, value->norm2);
	pq_sendfloat8(&buf, value->scale);
	pq_sendbytes(&buf, (char *) value->code, RABITQ_BYTES(value->dim, value->bits));
	return pq_endtypsend(&buf);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_in);
Datum
rabitq_in(PG_FUNCTION_ARGS)
{
	bytea	   *bytes = DatumGetByteaP(DirectFunctionCall1(byteain, PG_GETARG_DATUM(0)));
	StringInfoData buf;
	Rabitq	   *result;

	buf.data = VARDATA(bytes);
	buf.len = VARSIZE(bytes) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;
	result = Receive(&buf, PG_GETARG_INT32(2));
	pfree(bytes);
	PG_RETURN_POINTER(result);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_out);
Datum
rabitq_out(PG_FUNCTION_ARGS)
{
	Rabitq	   *value = PG_GETARG_RABITQ_P(0);
	bytea	   *bytes = Send(value);
	Datum		result = DirectFunctionCall1(byteaout, PointerGetDatum(bytes));

	pfree(bytes);
	PG_FREE_IF_COPY(value, 0);
	PG_RETURN_DATUM(result);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_recv);
Datum
rabitq_recv(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(Receive((StringInfo) PG_GETARG_POINTER(0), PG_GETARG_INT32(2)));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_send);
Datum
rabitq_send(PG_FUNCTION_ARGS)
{
	Rabitq	   *value = PG_GETARG_RABITQ_P(0);
	bytea	   *result = Send(value);

	PG_FREE_IF_COPY(value, 0);
	PG_RETURN_BYTEA_P(result);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_typmod_in);
Datum
rabitq_typmod_in(PG_FUNCTION_ARGS)
{
	int			n;
	int32	   *mods = ArrayGetIntegerTypmods(PG_GETARG_ARRAYTYPE_P(0), &n);

	if (n != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("rabitq requires one type modifier")));
	CheckDimensions(mods[0], -1);
	PG_RETURN_INT32(mods[0]);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_cast);
Datum
rabitq_cast(PG_FUNCTION_ARGS)
{
	Rabitq	   *value = PG_GETARG_RABITQ_P(0);

	CheckDimensions(value->dim, PG_GETARG_INT32(1));
	PG_RETURN_POINTER(value);
}

/* Used by index entry points to check dimensions against the metapage */
static Datum
RabitqDims(PG_FUNCTION_ARGS)
{
	Rabitq	   *value = PG_GETARG_RABITQ_P(0);

	PG_RETURN_INT32(value->dim);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_rabitq_support);
Datum
hnsw_rabitq_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = RABITQ_MAX_DIM,
		.dimensions = RabitqDims,
		.normalize = NULL,
		.checkValue = CheckValue,
		.useFsm = false
	};

	PG_RETURN_POINTER(&typeInfo);
}

/* The prepared query lives in the scan's resettable memory context. */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_prepare_query);
Datum
rabitq_prepare_query(PG_FUNCTION_ARGS)
{
	Vector	   *value = PG_GETARG_VECTOR_P(0);
	RabitqQuery *result;

	CheckDimensions(value->dim, -1);
	result = RabitqPrepareQuery(value->x, value->dim);
	PG_FREE_IF_COPY(value, 0);
	PG_RETURN_POINTER(result);
}

/* Cache scalar SQL's query by contents, never by Datum address. */
typedef struct QueryCache
{
	MemoryContext context;
	Vector	   *value;
	RabitqQuery *query;
} QueryCache;

static RabitqQuery *
GetQuery(FunctionCallInfo fcinfo, Vector *value)
{
	QueryCache *cache = fcinfo->flinfo->fn_extra;

	CheckDimensions(value->dim, -1);
	if (cache == NULL)
	{
		cache = MemoryContextAllocZero(fcinfo->flinfo->fn_mcxt, sizeof(QueryCache));
		cache->context = AllocSetContextCreate(fcinfo->flinfo->fn_mcxt,
											  "RaBitQ query", ALLOCSET_SMALL_SIZES);
		fcinfo->flinfo->fn_extra = cache;
	}
	if (cache->value == NULL || cache->value->dim != value->dim ||
		memcmp(cache->value->x, value->x, sizeof(float) * value->dim) != 0)
	{
		MemoryContext old;

		MemoryContextReset(cache->context);
		cache->value = NULL;
		cache->query = NULL;
		old = MemoryContextSwitchTo(cache->context);
		cache->query = RabitqPrepareQuery(value->x, value->dim);
		cache->value = palloc(VECTOR_SIZE(value->dim));
		memcpy(cache->value, value, VECTOR_SIZE(value->dim));
		MemoryContextSwitchTo(old);
	}
	return cache->query;
}

static double
L2Distance(double ip, double a2, double b2)
{
	return sqrt(Max(0.0, a2 + b2 - 2 * ip));
}

static double
QueryDistance(Rabitq *value, RabitqQuery *query)
{
	CheckVersion(value);
	CheckDimensions(query->dim, value->dim);
	return L2Distance(RabitqQueryIP(value, query), value->norm2, query->norm2);
}

static Datum
VectorDistance(FunctionCallInfo fcinfo, int valueArg, int queryArg)
{
	Rabitq	   *value = PG_GETARG_RABITQ_P(valueArg);
	Vector	   *query = PG_GETARG_VECTOR_P(queryArg);
	double		result = QueryDistance(value, GetQuery(fcinfo, query));

	PG_FREE_IF_COPY(value, valueArg);
	PG_FREE_IF_COPY(query, queryArg);
	PG_RETURN_FLOAT8(result);
}

/*
 * Get the estimated L2 distance between two codes, used to build the graph
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_l2_distance);
Datum
rabitq_l2_distance(PG_FUNCTION_ARGS)
{
	Rabitq	   *a = PG_GETARG_RABITQ_P(0);
	Rabitq	   *b = PG_GETARG_RABITQ_P(1);
	double		result;

	CheckVersion(a);
	CheckVersion(b);
	CheckDimensions(b->dim, a->dim);
	result = L2Distance(RabitqCodeIP(a, b), a->norm2, b->norm2);
	PG_FREE_IF_COPY(a, 0);
	PG_FREE_IF_COPY(b, 1);
	PG_RETURN_FLOAT8(result);
}

/*
 * Get the estimated L2 distance between a code and a vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_vector_l2_distance);
Datum
rabitq_vector_l2_distance(PG_FUNCTION_ARGS)
{
	return VectorDistance(fcinfo, 0, 1);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_rabitq_l2_distance);
Datum
vector_rabitq_l2_distance(PG_FUNCTION_ARGS)
{
	return VectorDistance(fcinfo, 1, 0);
}

/*
 * Get the estimated L2 distance between a prepared query and a code
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(rabitq_query_l2_distance);
Datum
rabitq_query_l2_distance(PG_FUNCTION_ARGS)
{
	RabitqQuery *query = (RabitqQuery *) PG_GETARG_POINTER(0);
	Rabitq	   *value = PG_GETARG_RABITQ_P(1);
	double		result = QueryDistance(value, query);

	PG_FREE_IF_COPY(value, 1);
	PG_RETURN_FLOAT8(result);
}
