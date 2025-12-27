use crate::cpu_hasher::{SimdExtension, hash_cpu, CpuTask, SafePointer};
use crate::buffer::PageAlignedByteBuffer;
#[cfg(feature = "opencl")]
use crate::gpu_hasher::{create_gpu_hasher_thread, GpuTask};
#[cfg(feature = "opencl")]
use crate::ocl::gpu_init;
use crate::hasher::{HasherTask, NONCE_SIZE};
#[cfg(feature = "opencl")]
use crossbeam_channel::unbounded;
use crossbeam_channel::{Receiver, Sender};
use std::cmp::min;
use std::sync::mpsc::channel;
use std::sync::Arc;
#[cfg(feature = "opencl")]
use std::thread;
use std::sync::atomic::{Ordering};

const CPU_TASK_SIZE: u64 = 64;

pub fn create_scheduler_thread(
    task: Arc<HasherTask>,
    thread_pool: rayon::ThreadPool,
    mut nonces_hashed: u64,
    pb: Option<indicatif::ProgressBar>,
    rx_empty_buffers: Receiver<PageAlignedByteBuffer>,
    tx_buffers_to_writer: Sender<PageAlignedByteBuffer>,
    simd_ext: SimdExtension,
) -> impl FnOnce() {
    move || {
        #[cfg(feature = "gui")]
        use crate::hasher::ProgressUpdate;
        
        #[cfg(feature = "gui")]
        let start_time = std::time::Instant::now();
        #[cfg(feature = "gui")]
        let mut last_speed_update_time = start_time;
        #[cfg(feature = "gui")]
        let mut total_nonces_processed = 0u64;

        // Get stop flag reference
        let stop_flag = task.stop_flag.clone();
        
        // Helper function to check stop
        let should_stop = || {
            stop_flag.as_ref().map_or(false, |flag| flag.load(Ordering::Relaxed))
        };

        // Check at the very beginning
        if should_stop() {
            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let _ = tx.send(ProgressUpdate::Log("Scheduler: Stop requested before starting".to_string()));
            }
            println!("Scheduler: Stop requested before starting");
            return;
        }

        let (tx, rx) = channel();

        #[cfg(feature = "opencl")]
        let gpu_contexts = match &task.gpus {
            Some(x) => Some(gpu_init(&x, task.zcb)),
            None => None,
        };

        #[cfg(feature = "opencl")]
        let gpus = match gpu_contexts {
            Some(x) => x,
            None => Vec::new(),
        };
        #[cfg(feature = "opencl")]
        let mut gpu_threads = Vec::new();
        #[cfg(feature = "opencl")]
        let mut gpu_channels = Vec::new();

        #[cfg(feature = "opencl")]
        for (i, gpu) in gpus.iter().enumerate() {
            gpu_channels.push(unbounded());
            gpu_threads.push(thread::spawn({
                create_gpu_hasher_thread(
                    (i + 1) as u8,
                    gpu.clone(),
                    tx.clone(),
                    gpu_channels.last().unwrap().1.clone(),
                )
            }));
        }

        // Simple buffer timing for logging purposes only
        let mut buffer_count: u32 = 0;
        let mut last_buffer_time = std::time::Instant::now();
        let mut avg_time_per_buffer = std::time::Duration::from_millis(0);

        while nonces_hashed < task.nonces && !should_stop() {
            // Check stop flag more frequently during long operations
            if buffer_count % 10 == 0 && should_stop() {
                #[cfg(feature = "gui")]
                if let Some(tx) = &task.progress_tx {
                    let _ = tx.send(ProgressUpdate::Log("Scheduler: Stop requested during processing".to_string()));
                }
                println!("Scheduler: Stop requested during processing");
                break;
            }
            
            buffer_count += 1;
            
            // Calculate time since last buffer for logging only
            let now = std::time::Instant::now();
            let time_since_last_buffer = now.duration_since(last_buffer_time);
            last_buffer_time = now;
            
            // Update rolling average for logging
            if avg_time_per_buffer == std::time::Duration::from_millis(0) {
                avg_time_per_buffer = time_since_last_buffer;
            } else {
                // Simple exponential smoothing for logging
                avg_time_per_buffer = std::time::Duration::from_nanos(
                    (avg_time_per_buffer.as_nanos() as f64 * 0.7 + 
                     time_since_last_buffer.as_nanos() as f64 * 0.3) as u64
                );
            }
            
            // Log buffer rate occasionally
            #[cfg(feature = "gui")]
            if let Some(tx_progress) = &task.progress_tx {
                if buffer_count % 200 == 0 {
                    let buffer_rate = 1000.0 / avg_time_per_buffer.as_millis() as f64;
                    let _ = tx_progress.send(ProgressUpdate::Log(
                        format!("Buffer rate: {:.1}/sec, Avg time: {:?}", buffer_rate, avg_time_per_buffer)
                    ));
                }
            }

            // Receive buffer with timeout to check stop flag
            let buffer = match rx_empty_buffers.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(buf) => buf,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Check stop flag during timeout
                    if should_stop() {
                        #[cfg(feature = "gui")]
                        if let Some(tx) = &task.progress_tx {
                            let _ = tx.send(ProgressUpdate::Log("Scheduler: Stop requested while waiting for buffer".to_string()));
                        }
                        println!("Scheduler: Stop requested while waiting for buffer");
                        break;
                    }
                    continue;
                }
                Err(_) => {
                    // Channel closed
                    break;
                }
            };

            let mut_bs = &buffer.get_buffer();
            let mut bs = mut_bs.lock().unwrap();
            let buffer_size = (*bs).len() as u64;
            let nonces_to_hash = min(buffer_size / NONCE_SIZE, task.nonces - nonces_hashed);

            let mut requested = 0u64;
            let mut processed = 0u64;

            #[cfg(feature = "opencl")]
            for (i, gpu) in gpus.iter().enumerate() {
                let gpu = gpu.lock().unwrap();
                let task_size = min(gpu.worksize as u64, nonces_to_hash - requested);
                if task_size > 0 {
                    let _ = gpu_channels[i]
                        .0
                        .send(Some(GpuTask {
                            cache: SafePointer {
                                ptr: bs.as_mut_ptr(),
                            },
                            cache_size: buffer_size / NONCE_SIZE,
                            chunk_offset: requested,
                            numeric_id: task.numeric_id,
                            local_startnonce: task.start_nonce + nonces_hashed + requested,
                            local_nonces: task_size,
                        }));
                }
                requested += task_size;
            }

            for _ in 0..task.cpu_threads {
                let task_size = min(CPU_TASK_SIZE, nonces_to_hash - requested);
                if task_size > 0 {
                    let task = hash_cpu(
                        tx.clone(),
                        CpuTask {
                            cache: SafePointer {
                                ptr: bs.as_mut_ptr(),
                            },
                            cache_size: (buffer_size / NONCE_SIZE) as usize,
                            chunk_offset: requested as usize,
                            numeric_id: task.numeric_id,
                            local_startnonce: task.start_nonce + nonces_hashed + requested,
                            local_nonces: task_size,
                        },
                        simd_ext.clone(),
                    );
                    thread_pool.spawn(task);
                }
                requested += task_size;
            }

            let rx = &rx;
            for msg in rx {
                // Check stop flag during message processing
                if should_stop() {
                    #[cfg(feature = "gui")]
                    if let Some(tx) = &task.progress_tx {
                        let _ = tx.send(ProgressUpdate::Log("Scheduler: Stop requested during hashing".to_string()));
                    }
                    println!("Scheduler: Stop requested during hashing");
                    break;
                }
                
                match msg.1 {
                    1 => {
                        let task_size = match msg.0 {
                            0 => {
                                let task_size = min(CPU_TASK_SIZE, nonces_to_hash - requested);
                                if task_size > 0 {
                                    let task = hash_cpu(
                                        tx.clone(),
                                        CpuTask {
                                            cache: SafePointer {
                                                ptr: bs.as_mut_ptr(),
                                            },
                                            cache_size: (buffer_size / NONCE_SIZE) as usize,
                                            chunk_offset: requested as usize,
                                            numeric_id: task.numeric_id,
                                            local_startnonce: task.start_nonce
                                                + nonces_hashed
                                                + requested,
                                            local_nonces: task_size,
                                        },
                                        simd_ext.clone(),
                                    );
                                    thread_pool.spawn(task);
                                }
                                task_size
                            }
                            _ => {
                                #[cfg(feature = "opencl")]
                                let gpu = gpus[(msg.0 - 1) as usize].lock().unwrap();
                                #[cfg(feature = "opencl")]
                                let task_size =
                                    min(gpu.worksize as u64, nonces_to_hash - requested);

                                #[cfg(feature = "opencl")]
                                let task_size = if task_size < gpu.worksize as u64
                                    && task.cpu_threads > 0
                                    && task_size > CPU_TASK_SIZE
                                {
                                    task_size / 2
                                } else {
                                    task_size
                                };

                                #[cfg(not(feature = "opencl"))]
                                let task_size = 0;

                                #[cfg(feature = "opencl")]
                                let _ = gpu_channels[(msg.0 - 1) as usize]
                                    .0
                                    .send(Some(GpuTask {
                                        cache: SafePointer {
                                            ptr: bs.as_mut_ptr(),
                                        },
                                        cache_size: buffer_size / NONCE_SIZE,
                                        chunk_offset: requested,
                                        numeric_id: task.numeric_id,
                                        local_startnonce: task.start_nonce
                                            + nonces_hashed
                                            + requested,
                                        local_nonces: task_size,
                                    }));
                                task_size
                            }
                        };

                        requested += task_size;
                    }
                    0 => {
                        processed += msg.2;
                        if let Some(pb) = &pb {
                            pb.inc(msg.2 * NONCE_SIZE);
                        }
                        

                        #[cfg(feature = "gui")]
                        if let Some(tx) = &task.progress_tx {
                            total_nonces_processed += msg.2;
                            

                            let current_nonces = nonces_hashed + processed;
                            let progress_pct = current_nonces as f32 / task.nonces as f32;
                            let _ = tx.send(ProgressUpdate::Progress(progress_pct));
                            

                            let now = std::time::Instant::now();
                            if now.duration_since(last_speed_update_time).as_secs() >= 1 {
                                let elapsed = now.duration_since(start_time).as_secs_f64();
                                if elapsed > 0.0 {
                                    let speed = total_nonces_processed as f64 * 60.0 / elapsed;
                                    let _ = tx.send(ProgressUpdate::Speed(speed));
                                }
                                last_speed_update_time = now;
                            }
                        }
                    }
                    _ => {}
                }
                if processed == nonces_to_hash {
                    break;
                }
            }

            // Check again before sending buffer to writer
            if should_stop() {
                #[cfg(feature = "gui")]
                if let Some(tx) = &task.progress_tx {
                    let _ = tx.send(ProgressUpdate::Log("Scheduler: Stop requested before sending buffer to writer".to_string()));
                }
                println!("Scheduler: Stop requested before sending buffer to writer");
                // Return buffer to pool
                let _ = tx_buffers_to_writer.send(buffer);
                break;
            }

            nonces_hashed += nonces_to_hash;
            

            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let progress_pct = nonces_hashed as f32 / task.nonces as f32;
                let _ = tx.send(ProgressUpdate::Progress(progress_pct));
            }

            let _ = tx_buffers_to_writer.send(buffer);

            if task.nonces == nonces_hashed {
                if let Some(pb) = &pb {
                    pb.finish_with_message("Hasher done.");
                }
                 #[cfg(feature = "gui")]
                if let Some(tx) = &task.progress_tx {
                    // Send final progress update
                    let _ = tx.send(ProgressUpdate::Progress(1.0));
                }


                #[cfg(feature = "opencl")]
                for gpu in &gpu_channels {
                    let _ = gpu.0.send(None);
                }
                break;
            }
        }
        
        // Cleanup: signal GPU threads to stop if we're exiting early
        if should_stop() {
            // Drop the tx channel to signal CPU/GPU workers that we're done
            drop(tx);
            
            #[cfg(feature = "opencl")]
            for gpu in &gpu_channels {
                let _ = gpu.0.send(None);
            }
        }
        
        // Wait for GPU threads
        #[cfg(feature = "opencl")]
        for thread in gpu_threads {
            let _ = thread.join();
        }
        
        #[cfg(feature = "gui")]
        if should_stop() && let Some(tx) = &task.progress_tx {
            let _ = tx.send(ProgressUpdate::Log("Scheduler: Exiting due to stop request".to_string()));
        }
        
        // Print to console when exiting due to stop (even in non-GUI mode)
        if should_stop() {
            println!("Scheduler: Exiting due to stop request");
        }
    }
}