use crate::hasher::{HasherTask, NONCE_SIZE, SCOOP_SIZE, NUM_SCOOPS};
use crate::buffer::PageAlignedByteBuffer;
use crate::utils::{open, open_r, open_using_direct_io};
use crossbeam_channel::{Receiver, Sender};
use std::cmp::min;
use std::io::{Read, Seek, SeekFrom, Write, Error, ErrorKind};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use indicatif::ProgressBar;
#[cfg(feature = "gui")]
use crate::hasher::ProgressUpdate;
use std::sync::atomic::{Ordering};

const TASK_SIZE: u64 = 16384;

pub fn create_writer_thread(
    task: Arc<HasherTask>,
    mut nonces_written: u64,
    pb: Option<ProgressBar>,
    rx_buffers_to_writer: Receiver<PageAlignedByteBuffer>,
    tx_empty_buffers: Sender<PageAlignedByteBuffer>,
) -> impl FnOnce() {
    move || {
        let mut last_speed_update_time = Instant::now();
        let mut bytes_written_since_last_update = 0u64;
        let mut file_bytes_written = nonces_written * NONCE_SIZE;
        let total_file_bytes = task.nonces * NONCE_SIZE;
        
        // Get stop flag
        let stop_flag = task.stop_flag.clone();
        let should_stop = || {
            stop_flag.as_ref().map_or(false, |flag| flag.load(Ordering::Relaxed))
        };

        #[cfg(feature = "gui")]
        let mut last_sent_percent = 0u32;
        
        // Check stop flag before starting
        if should_stop() {
            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let _ = tx.send(ProgressUpdate::Log("Writer: Stop requested before starting".to_string()));
            }
            println!("Writer: Stop requested before starting");
            return;
        }
        
        for buffer in rx_buffers_to_writer.iter() {
            // Check stop flag at the beginning of buffer processing
            if should_stop() {
                #[cfg(feature = "gui")]
                if let Some(tx) = &task.progress_tx {
                    let _ = tx.send(ProgressUpdate::Log("Writer: Stop requested".to_string()));
                }
                println!("Writer: Stop requested");
                // Return buffer to pool
                let _ = tx_empty_buffers.send(buffer);
                break;
            }
            
            let buffer_size;
            let nonces_to_write;
            
            {
                let mut_bs = &buffer.get_buffer();
                let bs = mut_bs.lock().unwrap();
                buffer_size = (*bs).len() as u64;
                nonces_to_write = min(buffer_size / NONCE_SIZE, task.nonces - nonces_written);
                
                let filename = Path::new(&task.output_path).join(format!(
                    "{}_{}_{}",
                    task.numeric_id, task.start_nonce, task.nonces
                ));
                
                if !task.benchmark {
                    // Check if file still exists (might have been deleted by stop)
                    if !filename.exists() {
                        #[cfg(feature = "gui")]
                        if let Some(tx) = &task.progress_tx {
                            let _ = tx.send(ProgressUpdate::Log("Writer: File deleted, stopping...".to_string()));
                        }
                        println!("Writer: File deleted, stopping...");
                        let _ = tx_empty_buffers.send(buffer);
                        break;
                    }
                
                    let file_result = if task.direct_io {
                        open_using_direct_io(&filename)
                    } else {
                        open(&filename)
                    };

                    let mut file: std::fs::File = match file_result {
                        Ok(f) => f,
                        Err(e) if e.raw_os_error() == Some(libc::EINVAL as i32) => {
                            match open(&filename) {
                                Ok(f) => f,
                                Err(e2) => {
                                    eprintln!("Error: Normal open also failed: {}", e2);
                                    tx_empty_buffers.send(buffer).unwrap();
                                    continue;
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Error: File open failed: {}", e);
                            tx_empty_buffers.send(buffer).unwrap();
                            continue;
                        }
                    };

                    let mut scoop_counter = 0u64;
                    let mut bytes_in_batch = 0u64;
                    
                    for scoop in 0..NUM_SCOOPS {
                        // Check stop flag during scoop processing
                        if scoop_counter % 16 == 0 && should_stop() {
                            #[cfg(feature = "gui")]
                            if let Some(tx) = &task.progress_tx {
                                let _ = tx.send(ProgressUpdate::Log(format!("Writer: Stop requested during scoop {}", scoop)));
                            }
                            println!("Writer: Stop requested during scoop {}", scoop);
                            break;
                        }
                        
                        let mut seek_addr = scoop * task.nonces as u64 * SCOOP_SIZE;
                        seek_addr += nonces_written as u64 * SCOOP_SIZE;

                        if let Err(e) = file.seek(SeekFrom::Start(seek_addr)) {
                            eprintln!("Seek failed for scoop {}: {}. Skipping scoop.", scoop, e);
                            continue;
                        }

                        let mut local_addr = scoop * buffer_size / NONCE_SIZE * SCOOP_SIZE;
                        

                        for _chunk in 0..(nonces_to_write / TASK_SIZE) {
                            // Check stop flag during chunk writing
                            if should_stop() {
                                #[cfg(feature = "gui")]
                                if let Some(tx) = &task.progress_tx {
                                    let _ = tx.send(ProgressUpdate::Log("Writer: Stop requested during chunk write".to_string()));
                                }
                                println!("Writer: Stop requested during chunk write");
                                break;
                            }
                            
                            let write_start = local_addr as usize;
                            let write_end = (local_addr + TASK_SIZE * SCOOP_SIZE) as usize;
                            
                            if let Err(e) = file.write_all(&bs[write_start..write_end]) {
                                eprintln!("Write failed in scoop {}: {}. Skipping chunk.", scoop, e);
                                break;
                            }
                            local_addr += TASK_SIZE * SCOOP_SIZE;
                            let chunk_bytes = TASK_SIZE * SCOOP_SIZE;
                            file_bytes_written += chunk_bytes;
                            bytes_written_since_last_update += chunk_bytes;
                            bytes_in_batch += chunk_bytes;
                        }

                        if !should_stop() && nonces_to_write % TASK_SIZE > 0 {
                            let write_start = local_addr as usize;
                            let write_end = (local_addr + (nonces_to_write % TASK_SIZE) * SCOOP_SIZE) as usize;
                            
                            if let Err(e) = file.write_all(&bs[write_start..write_end]) {
                                eprintln!("Remainder write failed in scoop {}: {}. Skipping.", scoop, e);
                            } else {
                                let remainder_bytes = (nonces_to_write % TASK_SIZE) * SCOOP_SIZE;
                                file_bytes_written += remainder_bytes;
                                bytes_written_since_last_update += remainder_bytes;
                                bytes_in_batch += remainder_bytes;
                            }
                        }

                        scoop_counter += 1;
                        

                        if scoop_counter % 64 == 0 {
                            if let Some(pb_ref) = &pb {
                                pb_ref.inc(bytes_in_batch);
                                bytes_in_batch = 0;
                            }
                        }
                        

                        #[cfg(feature = "gui")]
                        if let Some(tx) = &task.progress_tx {
                            let current_progress = file_bytes_written as f32 / total_file_bytes as f32;
                            let current_percent = (current_progress * 100.0).floor() as u32;
                            

                            if current_percent != last_sent_percent && current_percent <= 100 {
                                let _ = tx.try_send(ProgressUpdate::WriteProgress(current_progress.min(1.0)));
                                last_sent_percent = current_percent;
                            }
                        }
                        
                        // Check stop flag after scoop
                        if should_stop() {
                            break;
                        }
                    }
                    

                    if bytes_in_batch > 0 {
                        if let Some(pb_ref) = &pb {
                            pb_ref.inc(bytes_in_batch);
                            }
                    }
                        
                } else {

                    let buffer_bytes = nonces_to_write * NONCE_SIZE;
                    file_bytes_written += buffer_bytes;
                    bytes_written_since_last_update += buffer_bytes;
                    
                    if let Some(pb_ref) = &pb {
                        pb_ref.inc(buffer_bytes);
                    }
                }
            }
            
            nonces_written += nonces_to_write;

            if let Err(e) = tx_empty_buffers.send(buffer) {
                eprintln!("Warning: Could not return buffer to pool: {}", e);
            }

            if !task.benchmark && nonces_written > 0 && nonces_written % 10000 == 0 && !should_stop() {
                let filename = Path::new(&task.output_path).join(format!(
                    "{}_{}_{}",
                    task.numeric_id, task.start_nonce, task.nonces
                ));
                if write_resume_info(&filename, nonces_written).is_err() {
                    eprintln!("Warning: couldn't write resume info");
                }
            }

            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let elapsed = last_speed_update_time.elapsed();
                if elapsed >= Duration::from_secs(1) {
                    let speed_mbps = (bytes_written_since_last_update as f64 / elapsed.as_secs_f64()) / (1024.0 * 1024.0);
                    let _ = tx.try_send(ProgressUpdate::WriteSpeed(speed_mbps));
                    bytes_written_since_last_update = 0;
                    last_speed_update_time = Instant::now();
                }
            }

            if nonces_written == task.nonces {

                if !task.benchmark && !should_stop() {
                    let filename = Path::new(&task.output_path).join(format!(
                        "{}_{}_{}",
                        task.numeric_id, task.start_nonce, task.nonces
                    ));
                    if let Ok(file) = open(&filename) {
                        let _ = file.sync_all();
                    }
                }
                

                    if let Some(pb_ref) = &pb {
                        pb_ref.finish_with_message("Writer done.");
                    }
                    

                    #[cfg(feature = "gui")]
                    if let Some(tx) = &task.progress_tx {
                        // Send final write progress
                        let _ = tx.try_send(ProgressUpdate::WriteProgress(1.0f32));
                        
                        // Send completion log
                        let _ = tx.try_send(ProgressUpdate::Log("Writing completed".to_string()));
                    }
                    
                    if !task.benchmark && !should_stop() {
                        let filename = Path::new(&task.output_path).join(format!(
                            "{}_{}_{}",
                            task.numeric_id, task.start_nonce, task.nonces
                        ));
                        let _ = write_resume_info(&filename, nonces_written);
                    }
                
                break;
            }
            
            // Additional check after buffer processing
            if should_stop() {
                break;
            }
        }
        
        // Final cleanup if stopped
        if should_stop() {
            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let _ = tx.send(ProgressUpdate::Log("Writer: Exiting due to stop request".to_string()));
            }
            println!("Writer: Exiting due to stop request");
            
            // Try to sync file if it exists
            if !task.benchmark {
                let filename = Path::new(&task.output_path).join(format!(
                    "{}_{}_{}",
                    task.numeric_id, task.start_nonce, task.nonces
                ));
                if filename.exists() {
                    if let Ok(file) = open(&filename) {
                        let _ = file.sync_all();
                    }
                }
            }
        }
    }
}

pub fn read_resume_info(file: &Path) -> Result<u64, Error> {
    let mut file = open_r(&file)?;
    file.seek(SeekFrom::End(-8))?;

    let mut progress = [0u8; 4];
    let mut double_monkey = [0u8; 4];

    file.read_exact(&mut progress[0..4])?;
    file.read_exact(&mut double_monkey[0..4])?;

    if double_monkey == [0xAF, 0xFE, 0xAF, 0xFE] {
        Ok(u64::from(as_u32_le(progress)))
    } else {
        Err(Error::new(ErrorKind::Other, "End marker not found"))
    }
}

pub fn write_resume_info(file: &Path, nonces_written: u64) -> Result<(), Error> {
    let mut file = open(&file)?;
    file.seek(SeekFrom::End(-8))?;

    let progress = as_u8_le(nonces_written as u32);
    let double_monkey = [0xAF, 0xFE, 0xAF, 0xFE];

    file.write_all(&progress[0..4])?;
    file.write_all(&double_monkey[0..4])?;
    Ok(())    
}

fn as_u32_le(array: [u8; 4]) -> u32 {
    u32::from(array[0])
        + (u32::from(array[1]) << 8)
        + (u32::from(array[2]) << 16)
        + (u32::from(array[3]) << 24)
}

fn as_u8_le(x: u32) -> [u8; 4] {
    let b1: u8 = (x & 0xff) as u8;
    let b2: u8 = ((x >> 8) & 0xff) as u8;
    let b3: u8 = ((x >> 16) & 0xff) as u8;
    let b4: u8 = ((x >> 24) & 0xff) as u8;
    [b1, b2, b3, b4]
}