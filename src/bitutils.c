#include "postgres.h"

#include <math.h>

#include "bitutils.h"
#include "halfvec.h"			/* for USE_DISPATCH and USE_TARGET_CLONES */
#include "port/pg_bitutils.h"

#if defined(USE_DISPATCH)
#define BIT_DISPATCH
#endif

#ifdef BIT_DISPATCH
#include <immintrin.h>

#if defined(USE__GET_CPUID)
#include <cpuid.h>
#else
#include <intrin.h>
#endif

#ifdef _MSC_VER
#define TARGET_AVX512_POPCOUNT
#define TARGET_AVX512F
#else
#define TARGET_AVX512F __attribute__((target("avx512f")))
#define TARGET_AVX512_POPCOUNT __attribute__((target("avx512f,avx512vpopcntdq")))
#endif
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define BIT_NEON
#endif

/* Disable for LLVM due to crash with bitcode generation */
#if defined(USE_TARGET_CLONES) && !defined(__POPCNT__) && !defined(__llvm__)
#define BIT_TARGET_CLONES __attribute__((target_clones("default", "popcnt")))
#else
#define BIT_TARGET_CLONES
#endif

/* Use built-ins when possible for Postgres < 19 for inlining */
#if PG_VERSION_NUM >= 190000
#define popcount64(x) pg_popcount64(x)
#elif defined(HAVE__BUILTIN_POPCOUNT) && (defined(HAVE_LONG_INT_64) || SIZEOF_LONG == 8)
#define popcount64(x) __builtin_popcountl(x)
#elif defined(HAVE__BUILTIN_POPCOUNT) && (defined(HAVE_LONG_LONG_INT_64) || SIZEOF_LONG_LONG == 8)
#define popcount64(x) __builtin_popcountll(x)
#elif !defined(_MSC_VER)
/* Fails to resolve with MSVC */
#define popcount64(x) pg_popcount64(x)
#endif

uint64		(*BitHammingDistance) (uint32 bytes, unsigned char *ax, unsigned char *bx, uint64 distance);
double		(*BitJaccardDistance) (uint32 bytes, unsigned char *ax, unsigned char *bx, uint64 ab, uint64 aa, uint64 bb);

BIT_TARGET_CLONES static uint64
BitHammingDistanceDefault(uint32 bytes, unsigned char *ax, unsigned char *bx, uint64 distance)
{
#ifdef popcount64
	for (; bytes >= sizeof(uint64); bytes -= sizeof(uint64))
	{
		uint64		axs;
		uint64		bxs;

		/* Ensure aligned */
		memcpy(&axs, ax, sizeof(uint64));
		memcpy(&bxs, bx, sizeof(uint64));

		distance += popcount64(axs ^ bxs);

		ax += sizeof(uint64);
		bx += sizeof(uint64);
	}
#endif

	for (uint32 i = 0; i < bytes; i++)
		distance += pg_number_of_ones[ax[i] ^ bx[i]];

	return distance;
}

#ifdef BIT_DISPATCH
TARGET_AVX512_POPCOUNT static uint64
BitHammingDistanceAvx512Popcount(uint32 bytes, unsigned char *ax, unsigned char *bx, uint64 distance)
{
	__m512i		dist = _mm512_setzero_si512();

	for (; bytes >= sizeof(__m512i); bytes -= sizeof(__m512i))
	{
		__m512i		axs = _mm512_loadu_si512((const __m512i *) ax);
		__m512i		bxs = _mm512_loadu_si512((const __m512i *) bx);

		dist = _mm512_add_epi64(dist, _mm512_popcnt_epi64(_mm512_xor_si512(axs, bxs)));

		ax += sizeof(__m512i);
		bx += sizeof(__m512i);
	}

	distance += _mm512_reduce_add_epi64(dist);

	return BitHammingDistanceDefault(bytes, ax, bx, distance);
}
#endif

BIT_TARGET_CLONES static double
BitJaccardDistanceDefault(uint32 bytes, unsigned char *ax, unsigned char *bx, uint64 ab, uint64 aa, uint64 bb)
{
#ifdef popcount64
	for (; bytes >= sizeof(uint64); bytes -= sizeof(uint64))
	{
		uint64		axs;
		uint64		bxs;

		/* Ensure aligned */
		memcpy(&axs, ax, sizeof(uint64));
		memcpy(&bxs, bx, sizeof(uint64));

		ab += popcount64(axs & bxs);
		aa += popcount64(axs);
		bb += popcount64(bxs);

		ax += sizeof(uint64);
		bx += sizeof(uint64);
	}
#endif

	for (uint32 i = 0; i < bytes; i++)
	{
		ab += pg_number_of_ones[ax[i] & bx[i]];
		aa += pg_number_of_ones[ax[i]];
		bb += pg_number_of_ones[bx[i]];
	}

	if (ab == 0)
		return 1;
	else
		return 1 - (ab / ((double) (aa + bb - ab)));
}

#ifdef BIT_DISPATCH
TARGET_AVX512_POPCOUNT static double
BitJaccardDistanceAvx512Popcount(uint32 bytes, unsigned char *ax, unsigned char *bx, uint64 ab, uint64 aa, uint64 bb)
{
	__m512i		abx = _mm512_setzero_si512();
	__m512i		aax = _mm512_setzero_si512();
	__m512i		bbx = _mm512_setzero_si512();

	for (; bytes >= sizeof(__m512i); bytes -= sizeof(__m512i))
	{
		__m512i		axs = _mm512_loadu_si512((const __m512i *) ax);
		__m512i		bxs = _mm512_loadu_si512((const __m512i *) bx);

		abx = _mm512_add_epi64(abx, _mm512_popcnt_epi64(_mm512_and_si512(axs, bxs)));
		aax = _mm512_add_epi64(aax, _mm512_popcnt_epi64(axs));
		bbx = _mm512_add_epi64(bbx, _mm512_popcnt_epi64(bxs));

		ax += sizeof(__m512i);
		bx += sizeof(__m512i);
	}

	ab += _mm512_reduce_add_epi64(abx);
	aa += _mm512_reduce_add_epi64(aax);
	bb += _mm512_reduce_add_epi64(bbx);

	return BitJaccardDistanceDefault(bytes, ax, bx, ab, aa, bb);
}
#endif

/*
 * Varbit stores the first dimension in the most significant bit of a byte.
 * The sign table normalizes that order to increasing lanes.
 */
static uint32 BitVectorSignMask[256][BITS_PER_BYTE];

static double
BitVectorNegativeInnerProductDefault(const bits8 *bits, const float *values, int dimensions)
{
	double		distance = 0.0;

	for (int i = 0; i < dimensions; i++)
	{
		union
		{
			float		f;
			uint32		i;
		}			swapfloat;

		/* Flip the sign bit without a branch, since code bits are random */
		swapfloat.f = values[i];
		swapfloat.i ^= (uint32) ((bits[i / BITS_PER_BYTE] >> (BITS_PER_BYTE - 1 - (i % BITS_PER_BYTE))) & 1) << 31;
		distance += swapfloat.f;
	}

	return distance;
}

#ifdef BIT_DISPATCH
TARGET_AVX512F static double
BitVectorNegativeInnerProductAvx512(const bits8 *bits, const float *values, int dimensions)
{
	__m512		sum = _mm512_setzero_ps();
	int			i = 0;
	double		result;

	for (; i + 16 <= dimensions; i += 16)
	{
		__m512		current = _mm512_loadu_ps(values + i);
		__m512i		signs = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) BitVectorSignMask[bits[i >> 3]])),
											   _mm256_loadu_si256((const __m256i *) BitVectorSignMask[bits[(i >> 3) + 1]]), 1);

		/* Flip the sign before adding, so the sum has one dependency */
		sum = _mm512_add_ps(sum, _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(current), signs)));
	}

	result = (double) _mm512_reduce_add_ps(sum) +
		BitVectorNegativeInnerProductDefault(bits + (i >> 3), values + i, dimensions - i);

	/* Float lanes can overflow on huge values, so fall back to doubles */
	return isfinite(result) ? result : BitVectorNegativeInnerProductDefault(bits, values, dimensions);
}
#endif

#ifdef BIT_NEON
static double
BitVectorNegativeInnerProductNeon(const bits8 *bits, const float *values, int dimensions)
{
	float32x4_t low = vdupq_n_f32(0.0f);
	float32x4_t high = vdupq_n_f32(0.0f);
	int			i = 0;
	double		result;

	for (; i + BITS_PER_BYTE <= dimensions; i += BITS_PER_BYTE)
	{
		const uint32 *signs = BitVectorSignMask[bits[i >> 3]];
		float32x4_t currentLow = vld1q_f32(values + i);
		float32x4_t currentHigh = vld1q_f32(values + i + 4);
		uint32x4_t	signLow = vld1q_u32(signs);
		uint32x4_t	signHigh = vld1q_u32(signs + 4);

		low = vaddq_f32(low, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(currentLow), signLow)));
		high = vaddq_f32(high, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(currentHigh), signHigh)));
	}

	result = (double) vaddvq_f32(vaddq_f32(low, high)) +
		BitVectorNegativeInnerProductDefault(bits + (i >> 3), values + i, dimensions - i);

	/* Float lanes can overflow on huge values, so fall back to doubles */
	return isfinite(result) ? result : BitVectorNegativeInnerProductDefault(bits, values, dimensions);
}
#endif

double		(*BitVectorNegativeInnerProduct) (const bits8 *bits, const float *values, int dimensions);

#ifdef BIT_DISPATCH
#define CPU_FEATURE_OSXSAVE         (1 << 27)	/* F1 ECX */
#define CPU_FEATURE_AVX512F         (1 << 16)	/* F7,0 EBX */
#define CPU_FEATURE_AVX512VPOPCNTDQ (1 << 14)	/* F7,0 ECX */

#ifdef _MSC_VER
#define TARGET_XSAVE
#else
#define TARGET_XSAVE __attribute__((target("xsave")))
#endif

TARGET_XSAVE static bool
SupportsAvx512Popcount(void)
{
	unsigned int exx[4] = {0, 0, 0, 0};

#if defined(USE__GET_CPUID)
	__get_cpuid(1, &exx[0], &exx[1], &exx[2], &exx[3]);
#else
	__cpuid(exx, 1);
#endif

	/* Check OS supports XSAVE */
	if ((exx[2] & CPU_FEATURE_OSXSAVE) != CPU_FEATURE_OSXSAVE)
		return false;

	/* Check XMM, YMM, and ZMM registers are enabled */
	if ((_xgetbv(0) & 0xe6) != 0xe6)
		return false;

#if defined(USE__GET_CPUID)
	__get_cpuid_count(7, 0, &exx[0], &exx[1], &exx[2], &exx[3]);
#else
	__cpuidex(exx, 7, 0);
#endif

	/* Check AVX512F */
	if ((exx[1] & CPU_FEATURE_AVX512F) != CPU_FEATURE_AVX512F)
		return false;

	/* Check AVX512VPOPCNTDQ */
	return (exx[2] & CPU_FEATURE_AVX512VPOPCNTDQ) == CPU_FEATURE_AVX512VPOPCNTDQ;
}
#endif

void
BitvecInit(void)
{
	/*
	 * Could skip pointer when single function, but no difference in
	 * performance
	 */
	BitHammingDistance = BitHammingDistanceDefault;
	BitJaccardDistance = BitJaccardDistanceDefault;
	BitVectorNegativeInnerProduct = BitVectorNegativeInnerProductDefault;

	for (int byte = 0; byte < 256; byte++)
		for (int bit = 0; bit < BITS_PER_BYTE; bit++)
			BitVectorSignMask[byte][bit] = (byte & (1 << (BITS_PER_BYTE - 1 - bit))) ? 0x80000000U : 0;

#ifdef BIT_DISPATCH
	if (SupportsAvx512Popcount())
	{
		BitHammingDistance = BitHammingDistanceAvx512Popcount;
		BitJaccardDistance = BitJaccardDistanceAvx512Popcount;
		BitVectorNegativeInnerProduct = BitVectorNegativeInnerProductAvx512;
	}
#endif

#ifdef BIT_NEON
	BitVectorNegativeInnerProduct = BitVectorNegativeInnerProductNeon;
#endif
}
