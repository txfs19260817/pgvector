#include "postgres.h"

#include "bitutils.h"
#include "bitvec.h"
#include "fmgr.h"
#include "utils/varbit.h"
#include "vector.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

/*
 * Allocate and initialize a new bit vector
 */
VarBit *
InitBitVector(int dim)
{
	VarBit	   *result;
	Size		size;

	size = VARBITTOTALLEN(dim);
	result = (VarBit *) palloc0(size);
	SET_VARSIZE(result, size);
	VARBITLEN(result) = dim;

	return result;
}

/*
 * Ensure same dimensions
 */
static inline void
CheckDims(VarBit *a, VarBit *b)
{
	if (VARBITLEN(a) != VARBITLEN(b))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different bit lengths %u and %u", VARBITLEN(a), VARBITLEN(b))));
}

/*
 * Get the negative inner product of a sign code and a vector
 */
static double
BitVectorDistance(VarBit *a, Vector * b)
{
	if (VARBITLEN(a) != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different dimensions %u and %d", VARBITLEN(a), b->dim)));

	return BitVectorNegativeInnerProduct(VARBITS(a), b->x, b->dim);
}

/*
 * Get the Hamming distance between two bit vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hamming_distance);
Datum
hamming_distance(PG_FUNCTION_ARGS)
{
	VarBit	   *a = PG_GETARG_VARBIT_P(0);
	VarBit	   *b = PG_GETARG_VARBIT_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8((double) BitHammingDistance(VARBITBYTES(a), VARBITS(a), VARBITS(b), 0));
}

/*
 * Get the negative inner product of a sign code (set bit is +1, clear bit
 * is -1) and an unquantized vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(bit_vector_negative_inner_product);
Datum
bit_vector_negative_inner_product(PG_FUNCTION_ARGS)
{
	VarBit	   *a = PG_GETARG_VARBIT_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_FLOAT8(BitVectorDistance(a, b));
}

/*
 * Same with the query first, as HNSW scans call it
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_bit_negative_inner_product);
Datum
vector_bit_negative_inner_product(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	VarBit	   *b = PG_GETARG_VARBIT_P(1);

	PG_RETURN_FLOAT8(BitVectorDistance(b, a));
}

/*
 * Get the Jaccard distance between two bit vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(jaccard_distance);
Datum
jaccard_distance(PG_FUNCTION_ARGS)
{
	VarBit	   *a = PG_GETARG_VARBIT_P(0);
	VarBit	   *b = PG_GETARG_VARBIT_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8(BitJaccardDistance(VARBITBYTES(a), VARBITS(a), VARBITS(b), 0, 0, 0));
}
