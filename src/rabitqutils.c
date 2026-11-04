/*
 * Extended RaBitQ's inner-product estimator, with the origin as center:
 *     <x,q> ~= (||x||^2 / <Rx,b>) <b,Rq>.
 * b has odd integer coordinates from -(2^bits-1) to +(2^bits-1).
 * See arXiv:2409.09913 and the authors' RaBitQ-Library estimator notes.
 *
 * The directional quantizer below uses a bounded alternating scale search;
 * it does not claim the paper's globally optimal enumeration. The fixed
 * structured rotation is orthogonal, but is not Haar-random: the paper's
 * probabilistic error bound must not be used for pruning with this format.
 */
#include "postgres.h"

#include <math.h>

#include "bitutils.h"
#include "rabitq.h"

/* Fixed, portable mixing; changing any rotation detail requires a version. */
static uint64
RabitqHash(uint64 value)
{
	value += UINT64CONST(0x9e3779b97f4a7c15);
	value = (value ^ (value >> 30)) * UINT64CONST(0xbf58476d1ce4e5b9);
	value = (value ^ (value >> 27)) * UINT64CONST(0x94d049bb133111eb);
	return value ^ (value >> 31);
}

/*
 * Three signed, permuted block-Hadamard rounds. Binary decomposition of D
 * supplies the blocks; each is orthonormal and no coordinates are padded.
 * Permuting between rounds mixes coordinates across unequal blocks.
 */
static void
RabitqRotate(double *values, int dimensions)
{
	for (int round = 0; round < 3; round++)
	{
		uint64		state = RabitqHash((uint64) round);
		int			base = 0;

		for (int i = dimensions - 1; i > 0; i--)
		{
			int			j;
			double		tmp;

			state = RabitqHash(state);
			j = state % (i + 1);
			tmp = values[i];
			values[i] = values[j];
			values[j] = tmp;
		}
		for (int i = 0; i < dimensions; i++)
		{
			state = RabitqHash(state);
			if (state & 1)
				values[i] = -values[i];
		}
		while (base < dimensions)
		{
			int			size = 1;
			double		scale;

			while (size <= (dimensions - base) / 2)
				size *= 2;
			for (int step = 1; step < size; step *= 2)
				for (int i = base; i < base + size; i += 2 * step)
					for (int j = 0; j < step; j++)
					{
						double		a = values[i + j];
						double		b = values[i + j + step];

						values[i + j] = a + b;
						values[i + j + step] = a - b;
					}
			scale = 1.0 / sqrt((double) size);
			for (int i = base; i < base + size; i++)
				values[i] *= scale;
			base += size;
		}
	}
}

static inline int
RabitqCode(const Rabitq *value, int i)
{
	int			perByte = 8 / value->bits;
	int			mask = (1 << value->bits) - 1;

	return 2 * ((value->code[i / perByte] >> ((i % perByte) * value->bits)) & mask) - mask;
}

Rabitq *
RabitqEncode(const float *values, int dimensions, int bits)
{
	Rabitq	   *result = palloc0(RABITQ_SIZE(dimensions, bits));
	double	   *rotated = palloc(sizeof(double) * dimensions);
	int			mask = (1 << bits) - 1;
	int			perByte = 8 / bits;
	double		maximum = 0;
	double		bestScore = -1;

	SET_VARSIZE(result, RABITQ_SIZE(dimensions, bits));
	result->dim = dimensions;
	result->bits = bits;
	result->version = RABITQ_VERSION;
	for (int i = 0; i < dimensions; i++)
	{
		rotated[i] = values[i];
		result->norm2 += (double) values[i] * values[i];
	}
	RabitqRotate(rotated, dimensions);
	for (int i = 0; i < dimensions; i++)
		maximum = Max(maximum, fabs(rotated[i]));
	if (maximum == 0)
	{
		pfree(rotated);
		return result;
	}

	/* Include the sign direction, then refine several starting ranges. */
	for (int trial = 0; trial < (bits == 1 ? 1 : 5); trial++)
	{
		double		step = trial == 0 ? maximum :
			(trial == 1 ? maximum : Min(maximum, trial * sqrt(result->norm2 / dimensions))) / mask;

		for (int iteration = 0; iteration < 12; iteration++)
		{
			double		dot = 0;
			double		codeNorm2 = 0;
			double		score;
			double		next;

			for (int i = 0; i < dimensions; i++)
			{
				double		x = fabs(rotated[i]);
				int			level = x >= mask * step ? mask : 2 * (int) (x / (2 * step)) + 1;

				dot += x * level;
				codeNorm2 += level * level;
			}
			score = dot * dot / codeNorm2;
			if (score > bestScore)
			{
				bestScore = score;
				result->scale = result->norm2 / dot;
				memset(result->code, 0, RABITQ_BYTES(dimensions, bits));
				for (int i = 0; i < dimensions; i++)
				{
					double		x = fabs(rotated[i]);
					int			level = x >= mask * step ? mask : 2 * (int) (x / (2 * step)) + 1;
					int			code = ((rotated[i] < 0 ? -level : level) + mask) / 2;

					result->code[i / perByte] |= code << ((i % perByte) * bits);
				}
			}
			next = dot / codeNorm2;
			if (bits == 1 || fabs(next - step) <= step * 1e-7)
				break;
			step = next;
		}
	}
	pfree(rotated);
	return result;
}

RabitqQuery *
RabitqPrepareQuery(const float *values, int dimensions)
{
	RabitqQuery *result = palloc0(sizeof(RabitqQuery));
	int			groups = (dimensions + 3) / 4;

	result->dim = dimensions;
	/* Zero tail is query scratch only, never persisted code padding. */
	result->rotated = palloc0(sizeof(double) * groups * 4);
	result->lut = palloc(sizeof(double) * groups * 16);
	for (int i = 0; i < dimensions; i++)
	{
		result->rotated[i] = values[i];
		result->norm2 += (double) values[i] * values[i];
	}
	RabitqRotate(result->rotated, dimensions);
	for (int group = 0; group < groups; group++)
		for (int mask = 0; mask < 16; mask++)
		{
			double		sum = 0;

			for (int j = 0; j < 4; j++)
				sum += (mask & (1 << j) ? 1 : -1) * result->rotated[4 * group + j];
			result->lut[16 * group + mask] = sum;
		}
	return result;
}

double
RabitqQueryIP(const Rabitq *value, const RabitqQuery *query)
{
	double		sum = 0;

	if (value->bits == 1)
	{
		for (int i = 0; i < (value->dim + 3) / 4; i++)
		{
			int			mask = (value->code[i / 2] >> (4 * (i % 2))) & 15;

			sum += query->lut[i * 16 + mask];
		}
	}
	else if (value->bits == 4)
	{
		int			i = 0;

		for (; i + 1 < value->dim; i += 2)
		{
			unsigned	code = value->code[i / 2];

			sum += (2 * (int) (code & 15) - 15) * query->rotated[i];
			sum += (2 * (int) (code >> 4) - 15) * query->rotated[i + 1];
		}
		if (i < value->dim)
			sum += (2 * (int) (value->code[i / 2] & 15) - 15) * query->rotated[i];
	}
	else
	{
		for (int i = 0; i < value->dim; i++)
			sum += RabitqCode(value, i) * query->rotated[i];
	}
	return value->scale * sum;
}

double
RabitqCodeIP(const Rabitq *a, const Rabitq *b)
{
	/*
	 * Graph distance: rebuild each code at its true norm. The query factor
	 * (scale) would inflate a code's inner product with itself.
	 */
	int32		ab = 0;
	int32		aa = 0;
	int32		bb = 0;

	if (a->bits == 1 && b->bits == 1)
	{
		ab = a->dim - 2 * (int) BitHammingDistance(RABITQ_BYTES(a->dim, 1),
												(unsigned char *) a->code, (unsigned char *) b->code, 0);
		aa = bb = a->dim;
	}
	else if (a->bits == 4 && b->bits == 4)
	{
		int			i = 0;

		for (; i + 1 < a->dim; i += 2)
		{
			unsigned	ax = a->code[i / 2];
			unsigned	bx = b->code[i / 2];
			int			a0 = 2 * (int) (ax & 15) - 15;
			int			a1 = 2 * (int) (ax >> 4) - 15;
			int			b0 = 2 * (int) (bx & 15) - 15;
			int			b1 = 2 * (int) (bx >> 4) - 15;

			ab += a0 * b0 + a1 * b1;
			aa += a0 * a0 + a1 * a1;
			bb += b0 * b0 + b1 * b1;
		}
		if (i < a->dim)
		{
			ab += RabitqCode(a, i) * RabitqCode(b, i);
			aa += RabitqCode(a, i) * RabitqCode(a, i);
			bb += RabitqCode(b, i) * RabitqCode(b, i);
		}
	}
	else
	{
		for (int i = 0; i < a->dim; i++)
		{
			ab += RabitqCode(a, i) * RabitqCode(b, i);
			aa += RabitqCode(a, i) * RabitqCode(a, i);
			bb += RabitqCode(b, i) * RabitqCode(b, i);
		}
	}

	/* Levels are odd, so aa and bb are at least dim */
	return sqrt(a->norm2 * b->norm2 / ((double) aa * bb)) * ab;
}
