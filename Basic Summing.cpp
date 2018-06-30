#include <immintrin.h>
#include <LTBWindows.h>
#include <LTBCRT.h>
#define LTB_Types
#include <LTB.h>

/*
A simple experiment to see how much faster AVX can be
for a basic sum of array.
*/

int main() {

	enum : u64 {Length = 1 << 25, Scale = 256/32};
	i32* A = LTBAlloc(i32, Length);
	__m256i * AP = (__m256i*) A;
	LARGE_INTEGER QPC, QPF;
	QueryPerformanceFrequency(&QPF);
	f64 ClockFreq = (f64) QPF.QuadPart;
	u64 Cycles;

	for (u64 i = 0; i < Length; i++) {
		A[i] = 1;
	}

	enum : u64 {Repeats = 100};
	f64 AVXTimes[Repeats], UnrollTimes[Repeats];
	for (u64 R = 0; R < Repeats; R++) {

		QueryPerformanceCounter(&QPC);
		Cycles = QPC.QuadPart;
		enum : u64 {AVXUnrollFactor = 8}; // Found that 8 gives best result
		__m256i Sum[AVXUnrollFactor];
		for (u64 i = 0; i < 8; i++) {
			Sum[i] = _mm256_set1_epi32(0);
		}
		for (u64 i = 0; i < ((Length/Scale) - 1); i += AVXUnrollFactor) {
			Sum[0] = _mm256_add_epi32(Sum[0], AP[i]);
			Sum[1] = _mm256_add_epi32(Sum[1], AP[i + 1]);
			Sum[2] = _mm256_add_epi32(Sum[2], AP[i + 2]);
			Sum[3] = _mm256_add_epi32(Sum[3], AP[i + 3]);
			Sum[4] = _mm256_add_epi32(Sum[4], AP[i + 4]);
			Sum[5] = _mm256_add_epi32(Sum[5], AP[i + 5]);
			Sum[6] = _mm256_add_epi32(Sum[6], AP[i + 6]);
			Sum[7] = _mm256_add_epi32(Sum[7], AP[i + 7]);
		}
		u64 FinalSum = 0;
		for (u64 i = 0; i < AVXUnrollFactor; i++) {
			for (u64 j = 0; j < 8; j++) {
				FinalSum += Sum[i].m256i_i32[j];
			}
		}
		QueryPerformanceCounter(&QPC);
		Cycles = QPC.QuadPart - Cycles;
		AVXTimes[R] = Cycles/ClockFreq;

		QueryPerformanceCounter(&QPC);
		Cycles = QPC.QuadPart;
		u64 RegularSumArray[8] = {};
		for (u64 i = 0; i < (Length - 1); i += 8) {
			RegularSumArray[0] += A[i];
			RegularSumArray[1] += A[i + 1];
			RegularSumArray[2] += A[i + 2];
			RegularSumArray[3] += A[i + 3];
			RegularSumArray[4] += A[i + 4];
			RegularSumArray[5] += A[i + 5];
			RegularSumArray[6] += A[i + 6];
			RegularSumArray[7] += A[i + 7];

		}
		u64 RegularSum = 0;
		for (u64 i = 0; i < 8; i++) {
			RegularSum += RegularSumArray[i];
		}
		QueryPerformanceCounter(&QPC);
		Cycles = QPC.QuadPart - Cycles;
		UnrollTimes[R] = Cycles/ClockFreq;
	}

	f64 AVXTime = F64Average(AVXTimes, Repeats);
	f64 UnrollTime = F64Average(UnrollTimes, Repeats);
	// AVX is roughly 3.13 +- 0.03 times faster : 12.1 ms for AVX and 37.9 ms for unrolling.
	// This on a low power cpu: Intel i5-4210U 1.7GHZ
	// However, this is also only in debug mode - /O2 flattens the gap down
	// to less than a ms - AVX: 12.6ms, unrolling: 13.7ms

	#ifdef Optimized
	printf("AVXTime:    %f\nUnrollTime: %f", AVXTime, UnrollTime);
	getchar();
	#endif
	
	return (u64)(AVXTime + UnrollTime);
}