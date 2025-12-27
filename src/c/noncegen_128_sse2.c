#include "noncegen_128_avx.h"
#include <immintrin.h>
#include <string.h>
#include "common.h"
#include "mshabal_128_sse2.h"
#include "sph_shabal.h"

static sph_shabal_context global_32;
static mshabal128_context global_128;
static mshabal128_context_fast global_128_fast;

void init_shabal_sse2() {
    sph_shabal256_init(&global_32);
    mshabal_init_sse2(&global_128, 256);
    global_128_fast.out_size = global_128.out_size;
    for (int i = 0; i < 176; i++) global_128_fast.state[i] = global_128.state[i];
    global_128_fast.Whigh = global_128.Whigh;
    global_128_fast.Wlow = global_128.Wlow;
}

void noncegen_sse2(char *cache, const size_t cache_size, const size_t chunk_offset,
                   const uint64_t numeric_id, const uint64_t local_startnonce,
                   const uint64_t local_nonces) {
    sph_shabal_context local_32;
    uint64_t nonce;
    size_t len;

    mshabal128_context_fast local_128_fast;
    uint64_t nonce1, nonce2, nonce3, nonce4;

    char seed[32];
    char term[32];
    char zero[32];

    write_seed(seed, numeric_id);
    write_term(term);
    memset(&zero[0], 0, 32);

    uint8_t* buffer = (uint8_t*)malloc(sizeof(uint8_t) * MSHABAL128_VECTOR_SIZE * NONCE_SIZE);
    uint8_t* final = (uint8_t*)malloc(sizeof(uint8_t) * MSHABAL128_VECTOR_SIZE * HASH_SIZE);

    union {
        mshabal_u32 words[16 * MSHABAL128_VECTOR_SIZE];
        __m128i data[16];
    } t1, t2, t3;

    for (int j = 0; j < 16 * MSHABAL128_VECTOR_SIZE / 2; j += MSHABAL128_VECTOR_SIZE) {
        size_t o = j;

        t1.words[j + 0] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 1] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 2] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 3] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 0 + 32] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 1 + 32] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 2 + 32] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 3 + 32] = *(mshabal_u32 *)(zero + o);

        t2.words[j + 0 + 32] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 1 + 32] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 2 + 32] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 3 + 32] = *(mshabal_u32 *)(seed + o);

        t3.words[j + 0] = *(mshabal_u32 *)(term + o);
        t3.words[j + 1] = *(mshabal_u32 *)(term + o);
        t3.words[j + 2] = *(mshabal_u32 *)(term + o);
        t3.words[j + 3] = *(mshabal_u32 *)(term + o);
        t3.words[j + 0 + 32] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 1 + 32] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 2 + 32] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 3 + 32] = *(mshabal_u32 *)(zero + o);
    }

       for (uint64_t n = 0; n < local_nonces;) {

        if (n + 4 <= local_nonces) {

            nonce1 = bswap_64((uint64_t)(local_startnonce + n + 0));
            nonce2 = bswap_64((uint64_t)(local_startnonce + n + 1));
            nonce3 = bswap_64((uint64_t)(local_startnonce + n + 2));
            nonce4 = bswap_64((uint64_t)(local_startnonce + n + 3));

            for (int j = 8; j < 16; j += MSHABAL128_VECTOR_SIZE) {
                size_t o = j - 8;

                t1.words[j + 0] = *(mshabal_u32 *)((char *)&nonce1 + o);
                t1.words[j + 1] = *(mshabal_u32 *)((char *)&nonce2 + o);
                t1.words[j + 2] = *(mshabal_u32 *)((char *)&nonce3 + o);
                t1.words[j + 3] = *(mshabal_u32 *)((char *)&nonce4 + o);
                t2.words[j + 0 + 32] = *(mshabal_u32 *)((char *)&nonce1 + o);
                t2.words[j + 1 + 32] = *(mshabal_u32 *)((char *)&nonce2 + o);
                t2.words[j + 2 + 32] = *(mshabal_u32 *)((char *)&nonce3 + o);
                t2.words[j + 3 + 32] = *(mshabal_u32 *)((char *)&nonce4 + o);
            }

            memcpy(&local_128_fast, &global_128_fast,
                   sizeof(global_128_fast));

            mshabal_hash_fast_sse2(
                &local_128_fast, NULL, &t1,
                &buffer[MSHABAL128_VECTOR_SIZE * (NONCE_SIZE - HASH_SIZE)], 16 >> 6);

            memcpy(&t2, &buffer[MSHABAL128_VECTOR_SIZE * (NONCE_SIZE - HASH_SIZE)],
                   MSHABAL128_VECTOR_SIZE * (HASH_SIZE));

            for (size_t i = NONCE_SIZE - HASH_SIZE; i > (NONCE_SIZE - HASH_CAP); i -= HASH_SIZE) {

                if (i % 64 == 0) {

                    mshabal_hash_fast_sse2(&local_128_fast, &buffer[i * MSHABAL128_VECTOR_SIZE],
                                              &t1,
                                              &buffer[(i - HASH_SIZE) * MSHABAL128_VECTOR_SIZE],
                                              (NONCE_SIZE + 16 - i) >> 6);
                } else {

                    mshabal_hash_fast_sse2(&local_128_fast, &buffer[i * MSHABAL128_VECTOR_SIZE],
                                              &t2,
                                              &buffer[(i - HASH_SIZE) * MSHABAL128_VECTOR_SIZE],
                                              (NONCE_SIZE + 16 - i) >> 6);
                }
            }

            for (size_t i = NONCE_SIZE - HASH_CAP; i > 0; i -= HASH_SIZE) {
                mshabal_hash_fast_sse2(&local_128_fast, &buffer[i * MSHABAL128_VECTOR_SIZE], &t3,
                                          &buffer[(i - HASH_SIZE) * MSHABAL128_VECTOR_SIZE],
                                          (HASH_CAP) >> 6);
            }

            mshabal_hash_fast_sse2(&local_128_fast, &buffer[0], &t1, &final[0],
                                      (NONCE_SIZE + 16) >> 6);

            __m128i F[8];
            for (int j = 0; j < 8; j++) F[j] = _mm_loadu_si128((__m128i *)final + j);

            for (int j = 0; j < 8 * 2 * HASH_CAP; j++)
                _mm_storeu_si128(
                    (__m128i *)buffer + j,
                    _mm_xor_si128(_mm_loadu_si128((__m128i *)buffer + j), F[j % 8]));

            for (int i = 0; i < NUM_SCOOPS * 2; i++) {
                for (int j = 0; j < 32; j += 4) {
                    for (int k = 0; k < MSHABAL128_VECTOR_SIZE; k += 1) {
                    memcpy(&cache[((i & 1) * (4095 - (i >> 1)) + ((i + 1) & 1) * (i >> 1)) *
                                      SCOOP_SIZE * cache_size +
                                  (n + k + chunk_offset) * SCOOP_SIZE + (i & 1) * 32 + j],
                           &buffer[(i * 32 + j) * MSHABAL128_VECTOR_SIZE + k * 4], 4);
                    }
                }
            }

            n += 4;
        } else {

            int8_t *xv = (int8_t *)&numeric_id;

            for (size_t i = 0; i < 8; i++) buffer[NONCE_SIZE + i] = xv[7 - i];

            nonce = local_startnonce + n;
            xv = (int8_t *)&nonce;

            for (size_t i = 8; i < 16; i++) buffer[NONCE_SIZE + i] = xv[15 - i];

            for (size_t i = NONCE_SIZE; i > 0; i -= HASH_SIZE) {
                memcpy(&local_32, &global_32, sizeof(global_32));
                ;
                if (i < NONCE_SIZE + 16 - HASH_CAP)
                    len = HASH_CAP;
                else
                    len = NONCE_SIZE + 16 - i;

                sph_shabal256(&local_32, &buffer[i], len);
                sph_shabal256_close(&local_32, &buffer[i - HASH_SIZE]);
            }

            memcpy(&local_32, &global_32, sizeof(global_32));
            sph_shabal256(&local_32, buffer, 16 + NONCE_SIZE);
            sph_shabal256_close(&local_32, final);

            for (size_t i = 0; i < NONCE_SIZE; i++) buffer[i] ^= (final[i % HASH_SIZE]);

            for (size_t i = 0; i < HASH_CAP; i++){
                memmove(&cache[i * cache_size * SCOOP_SIZE + (n + chunk_offset) * SCOOP_SIZE], &buffer[i * SCOOP_SIZE], HASH_SIZE);
                memmove(&cache[(4095-i) * cache_size * SCOOP_SIZE + (n + chunk_offset) * SCOOP_SIZE + 32], &buffer[i * SCOOP_SIZE + 32], HASH_SIZE);
            }
            n++;
        }
    }
    free(buffer);
    free(final);
}