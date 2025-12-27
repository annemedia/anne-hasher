use crate::shabal256::shabal256_fast;

const HASH_SIZE: usize = 32;
const HASH_CAP: usize = 4096;
const NUM_SCOOPS: usize = 4096;
const SCOOP_SIZE: usize = 64;
const NONCE_SIZE: usize = NUM_SCOOPS * SCOOP_SIZE;
const MESSAGE_SIZE: usize = 16;

pub fn noncegen_rust(
    cache: &mut [u8],
    cache_offset: usize,
    numeric_id: u64,
    local_startnonce: u64,
    local_nonces: u64,
) {
    let numeric_id: [u32; 2] = unsafe { std::mem::transmute(numeric_id.to_be()) };

    let mut buffer = [0u8; NONCE_SIZE];
    let mut final_buffer = [0u8; HASH_SIZE];

    let mut t1 = [0u32; MESSAGE_SIZE];
    t1[0..2].clone_from_slice(&numeric_id);
    t1[4] = 0x80;

    let mut t2 = [0u32; MESSAGE_SIZE];
    t2[8..10].clone_from_slice(&numeric_id);
    t2[12] = 0x80;

    let mut t3 = [0u32; MESSAGE_SIZE];
    t3[0] = 0x80;

    for n in 0..local_nonces {

        let nonce: [u32; 2] = unsafe { std::mem::transmute((local_startnonce + n).to_be()) };

        t1[2..4].clone_from_slice(&nonce);
        t2[10..12].clone_from_slice(&nonce);

        let hash = shabal256_fast(&[], &t1);

        buffer[NONCE_SIZE - HASH_SIZE..NONCE_SIZE].clone_from_slice(&hash);
        let hash = unsafe { std::mem::transmute::<[u8; 32], [u32; 8]>(hash) };

        t2[0..8].clone_from_slice(&hash);

        for i in (NONCE_SIZE - HASH_CAP + HASH_SIZE..=NONCE_SIZE - HASH_SIZE)
            .rev()
            .step_by(HASH_SIZE)
        {

            if i % 64 == 0 {

                let hash = &shabal256_fast(&buffer[i..NONCE_SIZE], &t1);
                buffer[i - HASH_SIZE..i].clone_from_slice(hash);
            } else {

                let hash = &shabal256_fast(&buffer[i..NONCE_SIZE], &t2);
                buffer[i - HASH_SIZE..i].clone_from_slice(hash);
            }
        }

        for i in (HASH_SIZE..=NONCE_SIZE - HASH_CAP).rev().step_by(HASH_SIZE) {
            let hash = &shabal256_fast(&buffer[i..i + HASH_CAP], &t3);
            buffer[i - HASH_SIZE..i].clone_from_slice(hash);
        }

        final_buffer.clone_from_slice(&shabal256_fast(&buffer[0..NONCE_SIZE], &t1));

        for i in 0..NONCE_SIZE {
            buffer[i] ^= final_buffer[i % HASH_SIZE];
        }

        let cache_size = cache.len() / NONCE_SIZE;
        for i in 0..NUM_SCOOPS {
            let offset = i * cache_size * SCOOP_SIZE + (n as usize + cache_offset) * SCOOP_SIZE;
            cache[offset..offset + HASH_SIZE]
                .clone_from_slice(&buffer[i * SCOOP_SIZE..i * SCOOP_SIZE + HASH_SIZE]);
            let mirror_offset = (4095 - i) * cache_size * SCOOP_SIZE
                + (n as usize + cache_offset) * SCOOP_SIZE
                + HASH_SIZE;
            cache[mirror_offset..mirror_offset + HASH_SIZE].clone_from_slice(
                &buffer[i * SCOOP_SIZE + HASH_SIZE..i * SCOOP_SIZE + 2 * HASH_SIZE],
            );
        }
    }
}
