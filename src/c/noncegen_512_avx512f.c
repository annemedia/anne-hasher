#include "noncegen_512_avx512f.h"
#include <immintrin.h>
#include <string.h>
#include "common.h"
#include "mshabal_512_avx512f.h"
#include "sph_shabal.h"

static sph_shabal_context global_32;
static mshabal512_context global_512;
static mshabal512_context_fast global_512_fast;

void init_shabal_avx512f() {
    sph_shabal256_init(&global_32);
    mshabal_init_avx512f(&global_512, 256);
    global_512_fast.out_size = global_512.out_size;
    for (int i = 0; i < 704; i++) global_512_fast.state[i] = global_512.state[i];
    global_512_fast.Whigh = global_512.Whigh;
    global_512_fast.Wlow = global_512.Wlow;
}

void noncegen_avx512f(char *cache, const size_t cache_size, const size_t chunk_offset,
                   const uint64_t numeric_id, const uint64_t local_startnonce,
                   const uint64_t local_nonces) {
    sph_shabal_context local_32;
    uint64_t nonce;
    size_t len;

    mshabal512_context_fast local_512_fast;
    uint64_t nonce1, nonce2, nonce3, nonce4, nonce5, nonce6, nonce7, nonce8, nonce9, nonce10, nonce11, nonce12, nonce13, nonce14, nonce15, nonce16;

    char seed[32];
    char term[32];
    char zero[32];

    uint8_t* buffer = (uint8_t*)malloc(sizeof(uint8_t) * MSHABAL512_VECTOR_SIZE * NONCE_SIZE);
    uint8_t* final = (uint8_t*)malloc(sizeof(uint8_t) * MSHABAL512_VECTOR_SIZE * HASH_SIZE);
    
    write_seed(seed, numeric_id);
    write_term(term);
    memset(&zero[0], 0, 32);

    union {
        mshabal_u32 words[16 * MSHABAL512_VECTOR_SIZE];
        __m512i data[16];
    } t1, t2, t3;

    for (int j = 0; j < 16 * MSHABAL512_VECTOR_SIZE / 2; j += MSHABAL512_VECTOR_SIZE) {
        size_t o = j / 4;

        t1.words[j + 0] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 1] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 2] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 3] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 4] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 5] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 6] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 7] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 8] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 9] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 10] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 11] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 12] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 13] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 14] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 15] = *(mshabal_u32 *)(seed + o);        
        t1.words[j + 0 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 1 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 2 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 3 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 4 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 5 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 6 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 7 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 8 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 9 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 10 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 11 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 12 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 13 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 14 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 15 + 128] = *(mshabal_u32 *)(zero + o);

        t2.words[j + 0 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 1 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 2 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 3 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 4 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 5 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 6 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 7 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 8 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 9 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 10 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 11 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 12 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 13 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 14 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 15 + 128] = *(mshabal_u32 *)(seed + o);

        t3.words[j + 0] = *(mshabal_u32 *)(term + o);
        t3.words[j + 1] = *(mshabal_u32 *)(term + o);
        t3.words[j + 2] = *(mshabal_u32 *)(term + o);
        t3.words[j + 3] = *(mshabal_u32 *)(term + o);
        t3.words[j + 4] = *(mshabal_u32 *)(term + o);
        t3.words[j + 5] = *(mshabal_u32 *)(term + o);
        t3.words[j + 6] = *(mshabal_u32 *)(term + o);
        t3.words[j + 7] = *(mshabal_u32 *)(term + o);        
        t3.words[j + 8] = *(mshabal_u32 *)(term + o);
        t3.words[j + 9] = *(mshabal_u32 *)(term + o);
        t3.words[j + 10] = *(mshabal_u32 *)(term + o);
        t3.words[j + 11] = *(mshabal_u32 *)(term + o);
        t3.words[j + 12] = *(mshabal_u32 *)(term + o);
        t3.words[j + 13] = *(mshabal_u32 *)(term + o);
        t3.words[j + 14] = *(mshabal_u32 *)(term + o);
        t3.words[j + 15] = *(mshabal_u32 *)(term + o);
        
        t3.words[j + 0 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 1 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 2 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 3 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 4 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 5 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 6 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 7 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 8 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 9 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 10 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 11 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 12 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 13 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 14 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 15 + 128] = *(mshabal_u32 *)(zero + o);
    }

    for (uint64_t n = 0; n < local_nonces;) {

        if (n + 16 <= local_nonces) {

            nonce1 = bswap_64((uint64_t)(local_startnonce + n + 0));
            nonce2 = bswap_64((uint64_t)(local_startnonce + n + 1));
            nonce3 = bswap_64((uint64_t)(local_startnonce + n + 2));
            nonce4 = bswap_64((uint64_t)(local_startnonce + n + 3));
            nonce5 = bswap_64((uint64_t)(local_startnonce + n + 4));
            nonce6 = bswap_64((uint64_t)(local_startnonce + n + 5));
            nonce7 = bswap_64((uint64_t)(local_startnonce + n + 6));
            nonce8 = bswap_64((uint64_t)(local_startnonce + n + 7));
            nonce9 = bswap_64((uint64_t)(local_startnonce + n + 8));
            nonce10 = bswap_64((uint64_t)(local_startnonce + n + 9));
            nonce11 = bswap_64((uint64_t)(local_startnonce + n + 10));
            nonce12 = bswap_64((uint64_t)(local_startnonce + n + 11));
            nonce13 = bswap_64((uint64_t)(local_startnonce + n + 12));
            nonce14 = bswap_64((uint64_t)(local_startnonce + n + 13));
            nonce15 = bswap_64((uint64_t)(local_startnonce + n + 14));
            nonce16 = bswap_64((uint64_t)(local_startnonce + n + 15));

            for (int j = 32; j < 16 * MSHABAL512_VECTOR_SIZE / 4; j += MSHABAL512_VECTOR_SIZE) {
                size_t o = j / 4 - 8;

                t1.words[j + 0] = *(mshabal_u32 *)((char *)&nonce1 + o);
                t1.words[j + 1] = *(mshabal_u32 *)((char *)&nonce2 + o);
                t1.words[j + 2] = *(mshabal_u32 *)((char *)&nonce3 + o);
                t1.words[j + 3] = *(mshabal_u32 *)((char *)&nonce4 + o);
                t1.words[j + 4] = *(mshabal_u32 *)((char *)&nonce5 + o);
                t1.words[j + 5] = *(mshabal_u32 *)((char *)&nonce6 + o);
                t1.words[j + 6] = *(mshabal_u32 *)((char *)&nonce7 + o);
                t1.words[j + 7] = *(mshabal_u32 *)((char *)&nonce8 + o);
                t1.words[j + 8] = *(mshabal_u32 *)((char *)&nonce9 + o);
                t1.words[j + 9] = *(mshabal_u32 *)((char *)&nonce10 + o);
                t1.words[j + 10] = *(mshabal_u32 *)((char *)&nonce11 + o);
                t1.words[j + 11] = *(mshabal_u32 *)((char *)&nonce12 + o);
                t1.words[j + 12] = *(mshabal_u32 *)((char *)&nonce13 + o);
                t1.words[j + 13] = *(mshabal_u32 *)((char *)&nonce14 + o);
                t1.words[j + 14] = *(mshabal_u32 *)((char *)&nonce15 + o);
                t1.words[j + 15] = *(mshabal_u32 *)((char *)&nonce16 + o);

                t2.words[j + 0 + 128] = *(mshabal_u32 *)((char *)&nonce1 + o);
                t2.words[j + 1 + 128] = *(mshabal_u32 *)((char *)&nonce2 + o);
                t2.words[j + 2 + 128] = *(mshabal_u32 *)((char *)&nonce3 + o);
                t2.words[j + 3 + 128] = *(mshabal_u32 *)((char *)&nonce4 + o);
                t2.words[j + 4 + 128] = *(mshabal_u32 *)((char *)&nonce5 + o);
                t2.words[j + 5 + 128] = *(mshabal_u32 *)((char *)&nonce6 + o);
                t2.words[j + 6 + 128] = *(mshabal_u32 *)((char *)&nonce7 + o);
                t2.words[j + 7 + 128] = *(mshabal_u32 *)((char *)&nonce8 + o); 
                t2.words[j + 8 + 128] = *(mshabal_u32 *)((char *)&nonce9 + o);
                t2.words[j + 9 + 128] = *(mshabal_u32 *)((char *)&nonce10 + o);
                t2.words[j + 10 + 128] = *(mshabal_u32 *)((char *)&nonce11 + o);
                t2.words[j + 11 + 128] = *(mshabal_u32 *)((char *)&nonce12 + o);
                t2.words[j + 12 + 128] = *(mshabal_u32 *)((char *)&nonce13 + o);
                t2.words[j + 13 + 128] = *(mshabal_u32 *)((char *)&nonce14 + o);
                t2.words[j + 14 + 128] = *(mshabal_u32 *)((char *)&nonce15 + o);
                t2.words[j + 15 + 128] = *(mshabal_u32 *)((char *)&nonce16 + o);
            }
    

            

            memcpy(&local_512_fast, &global_512_fast,
                   sizeof(global_512_fast));
            
             mshabal_hash_fast_avx512f(
                &local_512_fast, NULL, &t1,
                &buffer[MSHABAL512_VECTOR_SIZE * (NONCE_SIZE - HASH_SIZE)], 16 >> 6);

            memcpy(&t2, &buffer[MSHABAL512_VECTOR_SIZE * (NONCE_SIZE - HASH_SIZE)],
                   MSHABAL512_VECTOR_SIZE * (HASH_SIZE));

            for (size_t i = NONCE_SIZE - HASH_SIZE; i > (NONCE_SIZE - HASH_CAP); i -= HASH_SIZE) {

                if (i % 64 == 0) {

                     mshabal_hash_fast_avx512f(&local_512_fast, &buffer[i * MSHABAL512_VECTOR_SIZE],
                                              &t1,
                                              &buffer[(i - HASH_SIZE) * MSHABAL512_VECTOR_SIZE],
                                              (NONCE_SIZE + 16 - i) >> 6);
                } else {

                     mshabal_hash_fast_avx512f(&local_512_fast, &buffer[i * MSHABAL512_VECTOR_SIZE],
                                              &t2,
                                              &buffer[(i - HASH_SIZE) * MSHABAL512_VECTOR_SIZE],
                                              (NONCE_SIZE + 16 - i) >> 6);
                }
            }

            for (size_t i = NONCE_SIZE - HASH_CAP; i > 0; i -= HASH_SIZE) {
                 mshabal_hash_fast_avx512f(&local_512_fast, &buffer[i * MSHABAL512_VECTOR_SIZE], &t3,
                                          &buffer[(i - HASH_SIZE) * MSHABAL512_VECTOR_SIZE],
                                          (HASH_CAP) >> 6);
            }
           

             mshabal_hash_fast_avx512f(&local_512_fast, &buffer[0], &t1, &final[0],
                                      (NONCE_SIZE + 16) >> 6);
            

            __m512i F[8];
            for (int j = 0; j < 8; j++) F[j] = _mm512_loadu_si512((__m512i *)final + j);

            for (int j = 0; j < 8 * 2 * HASH_CAP; j++)
                _mm512_storeu_si512(
                    (__m512i *)buffer + j,
                    _mm512_xor_si512(_mm512_loadu_si512((__m512i *)buffer + j), F[j % 8]));
             

            for (int i = 0; i < NUM_SCOOPS * 2; i++) {
                for (int j = 0; j < 32; j += 4) {
                    for (int k = 0; k < MSHABAL512_VECTOR_SIZE; k += 1) {
                    memcpy(&cache[((i & 1) * (4095 - (i >> 1)) + ((i + 1) & 1) * (i >> 1)) *
                                      SCOOP_SIZE * cache_size +
                                  (n + k + chunk_offset) * SCOOP_SIZE + (i & 1) * 32 + j],
                           &buffer[(i * 32 + j) * MSHABAL512_VECTOR_SIZE + k * 4], 4);
                    }
                }
            }

            n += 16;
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
