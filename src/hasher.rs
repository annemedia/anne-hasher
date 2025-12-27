
use humanize_rs::bytes::Bytes;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
#[cfg(target_arch = "x86_64")]
use raw_cpuid::CpuId;

use crate::cpu_hasher::{SimdExtension, init_simd};
use crate::buffer::PageAlignedByteBuffer;
#[cfg(feature = "opencl")]
use crate::ocl::gpu_get_info;
use crate::scheduler::create_scheduler_thread;
#[cfg(windows)]
use crate::utils::set_thread_ideal_processor;
use crate::utils::{free_disk_space, get_sector_size, preallocate};
use crate::writer::{create_writer_thread, read_resume_info, write_resume_info};
use core_affinity;
use crossbeam_channel::bounded;
#[cfg(feature = "gui")]
use crossbeam_channel::Sender;
use std::cmp::{max, min};
use std::path::Path;
use std::process;

use std::thread;
use stopwatch::Stopwatch;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const SCOOP_SIZE: u64 = 64;
pub const NUM_SCOOPS: u64 = 4096;
pub const NONCE_SIZE: u64 = SCOOP_SIZE * NUM_SCOOPS;

#[cfg(feature = "gui")]
#[derive(Debug, Clone)]
pub enum ProgressUpdate {
    Log(String),
    Progress(f32),
    WriteProgress(f32),
    Speed(f64),
    WriteSpeed(f64),
    Error(String),
    Done
}

pub struct Hasher {}

pub struct HasherTask {
    pub numeric_id: u64,
    pub start_nonce: u64,
    pub nonces: u64,
    pub output_path: String,
    pub mem: String,
    pub cpu_threads: u8,
    pub gpus: Option<Vec<String>>,
    pub direct_io: bool,
    pub benchmark: bool,
    #[cfg(feature = "opencl")]
    pub zcb: bool,

    #[cfg(feature = "gui")]
    pub progress_tx: Option<Sender<ProgressUpdate>>,
    pub stop_flag: Option<Arc<AtomicBool>>,
}

impl Hasher {
    pub fn new() -> Hasher {
        Hasher {}
    }

    pub fn run(&self, mut task: HasherTask) {
        // let cpuid = CpuId::new();
        // let cpu_name = cpuid
        //     .get_processor_brand_string()
        //     .map(|s| s.as_str().trim().to_string())
        //     .unwrap_or_else(|| {
        //         cpuid
        //             .get_vendor_info()
        //             .map(|v| v.as_str().to_string())
        //             .unwrap_or_else(|| "Unknown CPU".to_string())
        //     });
        let cpu_name: String = {
            #[cfg(target_arch = "x86_64")]
            {
                let cpuid = CpuId::new();
                if let Some(pbs) = cpuid.get_processor_brand_string() {
                    pbs.as_str().trim().to_string()
                } else if let Some(vi) = cpuid.get_vendor_info() {
                    vi.as_str().to_string()
                } else {
                    "Unknown CPU".to_string()
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                "Apple Silicon (aarch64)".to_string()
            }
        };

        let cores = sys_info::cpu_num().unwrap();
        let memory = sys_info::mem_info().unwrap();

        let simd_ext = init_simd();

        #[cfg(feature = "gui")]
        if let Some(tx) = &task.progress_tx {
            let _ = tx.send(ProgressUpdate::Log(format!("anne-hasher {}\n", env!("CARGO_PKG_VERSION"))));
            if task.benchmark {
                let _ = tx.send(ProgressUpdate::Log("*BENCHMARK MODE*\n".to_string()));
            }
            let _ = tx.send(ProgressUpdate::Log(format!(
                "CPU: {} [using {} of {} cores{}{:?}]",
                cpu_name,
                task.cpu_threads,
                cores,
                if let SimdExtension::None = &simd_ext { "" } else { " + " },
                &simd_ext
            )));
        }

        println!("anne-hasher {}\n", env!("CARGO_PKG_VERSION"));
        if task.benchmark {
                println!("*BENCHMARK MODE*\n");
        }
        println!(
            "CPU: {} [using {} of {} cores{}{:?}]",
            cpu_name,
            task.cpu_threads,
            cores,
            if let SimdExtension::None = &simd_ext { "" } else { " + " },
            &simd_ext
        );

        #[cfg(not(feature = "opencl"))]
        let gpu_mem_needed = 0u64;
        #[cfg(feature = "opencl")]
        let gpu_mem_needed = match &task.gpus {
            Some(x) => gpu_get_info(&x),
            None => 0,
        };

        #[cfg(feature = "opencl")]
        let gpu_mem_needed = if task.zcb { gpu_mem_needed } else { gpu_mem_needed / 2 };

        let free_disk_space = free_disk_space(&task.output_path);
        if task.nonces == 0 {
            task.nonces = free_disk_space / NONCE_SIZE;
        }

        let gpu = task.gpus.is_some();

        let mut rounded_nonces_to_sector_size = false;
        let mut nonces_per_sector = 1;
        if task.direct_io {
            let sector_size = get_sector_size(&task.output_path);
            nonces_per_sector = sector_size / SCOOP_SIZE;
            if task.nonces % nonces_per_sector > 0 {
                rounded_nonces_to_sector_size = true;
                task.nonces /= nonces_per_sector;
                task.nonces *= nonces_per_sector;
            }
        }

        let segmentsize = task.nonces * NONCE_SIZE;

        let file = Path::new(&task.output_path).join(format!(
            "{}_{}_{}",
            task.numeric_id,
            task.start_nonce,
            task.nonces
        ));

        if !file.parent().unwrap().exists() {
            let msg = format!(
                "Error: specified target path does not exist, path={}",
                &task.output_path
            );
            println!("{}", msg);
            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let _ = tx.send(ProgressUpdate::Error(msg));
            }
            println!("Shutting down...");
            return;
        }

        if free_disk_space < segmentsize && !file.exists() && !task.benchmark {
            let msg = format!(
                "Error: insufficient disk space, MiB_required={:.2}, MiB_available={:.2}",
                segmentsize as f64 / 1024.0 / 1024.0,
                free_disk_space as f64 / 1024.0 / 1024.0
            );
            println!("{}", msg);
            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let _ = tx.send(ProgressUpdate::Error(msg));
            }
            println!("Shutting down...");
            return;
        }

        let mem = match calculate_mem_to_use(&task, &memory, nonces_per_sector, gpu, gpu_mem_needed) {
            Ok(x) => x,
            Err(_) => return,
        };

        #[cfg(feature = "gui")]
        if let Some(tx) = &task.progress_tx {
            let _ = tx.send(ProgressUpdate::Log(format!(
                "RAM: Total={:.2} GiB, Free={:.2} GiB, Usage={:.2} GiB",
                memory.total as f64 / 1024.0 / 1024.0,
                get_avail_mem(&memory) as f64 / 1024.0 / 1024.0,
                (mem + gpu_mem_needed) as f64 / 1024.0 / 1024.0 / 1024.0
            )));

            #[cfg(feature = "opencl")]
            let _ = tx.send(ProgressUpdate::Log(format!(
                "     HDDcache={:.2} GiB, GPUcache={:.2} GiB,\n",
                mem as f64 / 1024.0 / 1024.0 / 1024.0,
                gpu_mem_needed as f64 / 1024.0 / 1024.0 / 1024.0
            )));

            let _ = tx.send(ProgressUpdate::Log(format!("Numeric ID:  {}", task.numeric_id)));
            let _ = tx.send(ProgressUpdate::Log(format!("Start Nonce: {}", task.start_nonce)));
            let _ = tx.send(ProgressUpdate::Log(format!(
                "Nonces:      {}{}",
                task.nonces,
                if rounded_nonces_to_sector_size {
                    " (rounded to sector size for fast direct i/o)"
                } else {
                    ""
                }
            )));
            let _ = tx.send(ProgressUpdate::Log(format!("Output File: {}\n", file.display())));
        }

        println!(
                "RAM: Total={:.2} GiB, Free={:.2} GiB, Usage={:.2} GiB",
                memory.total as f64 / 1024.0 / 1024.0,
                get_avail_mem(&memory) as f64 / 1024.0 / 1024.0,
                (mem + gpu_mem_needed) as f64 / 1024.0 / 1024.0 / 1024.0
            );

            #[cfg(feature = "opencl")]
            println!(
                "     HDDcache={:.2} GiB, GPUcache={:.2} GiB,\n",
                mem as f64 / 1024.0 / 1024.0 / 1024.0,
                gpu_mem_needed as f64 / 1024.0 / 1024.0 / 1024.0
            );

            println!("Numeric ID:  {}", task.numeric_id);
            println!("Start Nonce: {}", task.start_nonce);
            println!(
                "Nonces:      {}{}",
                task.nonces,
                if rounded_nonces_to_sector_size {
                    " (rounded to sector size for fast direct i/o)"
                } else {
                    ""
                }
            );
            println!("Output File: {}\n", file.display());

        let mut progress = 0u64;
        if file.exists() {
            println!("File already exists, reading resume info...");
            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let _ = tx.send(ProgressUpdate::Log("File exists, reading resume info...".to_string()));
            }

            let resume_info = read_resume_info(&file);
            match resume_info {
                Ok(x) => progress = x,
                Err(_) => {
                    let msg = format!("Error: couldn't read resume info from file '{}'", file.display());
                    println!("{}", msg);
                    println!("If you are sure that this file is incomplete or corrupted, then delete it before continuing.");
                    println!("Shutting Down...");
                    #[cfg(feature = "gui")]
                    if let Some(tx) = &task.progress_tx {
                        let _ = tx.send(ProgressUpdate::Error(msg));
                    }
                    return;
                }
            }
            println!("OK");
            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let _ = tx.send(ProgressUpdate::Log("Resume info loaded.".to_string()));
            }
        } else {
            print!("Pre-allocating file, please wait...");
            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let _ = tx.send(ProgressUpdate::Log("Pre-allocating file, please wait...".to_string()));
            }
            if !task.benchmark {
                preallocate(&file, segmentsize, task.direct_io);
                if write_resume_info(&file, 0u64).is_err() {
                    println!("Error: couldn't write resume info");
                    #[cfg(feature = "gui")]
                    if let Some(tx) = &task.progress_tx {
                        let _ = tx.send(ProgressUpdate::Error("Failed to write resume info".to_string()));
                    }
                }
            }
            println!("OK");
            #[cfg(feature = "gui")]
            if let Some(tx) = &task.progress_tx {
                let _ = tx.send(ProgressUpdate::Log("Pre-allocation complete.".to_string()));
            }
        }

        if progress == 0 {
            println!("Starting hashing...\n");
        } else {
            println!("Resuming hashing from nonce offset {}...\n", progress);
        }
        #[cfg(feature = "gui")]
        if let Some(tx) = &task.progress_tx {
            if progress == 0 {
                let _ = tx.send(ProgressUpdate::Log("Starting hashing...\n".to_string()));
            } else {
                let _ = tx.send(ProgressUpdate::Log(format!("Resuming hashing from nonce offset {}...\n", progress)));
            }
        }

        let num_buffer = { 2 };
        let buffer_size = mem / num_buffer;
        let (tx_empty_buffers, rx_empty_buffers) = bounded(num_buffer as usize);
        let (tx_full_buffers, rx_full_buffers) = bounded(num_buffer as usize);

        for _ in 0..num_buffer {
            let buffer = PageAlignedByteBuffer::new(buffer_size as usize);
            tx_empty_buffers.send(buffer).unwrap();
        }

        let mb = MultiProgress::new();

        let p1x = {
            let pb = mb.add(ProgressBar::new(segmentsize - progress * NONCE_SIZE));
            pb.set_style(ProgressStyle::default_bar()
                .template("{prefix:>12} {wide_bar} {bytes:>8} {bytes_per_sec:>10}")
                .expect("Failed to set template")
                .progress_chars("██░"));
            pb.set_prefix("Hashing:");
            pb.enable_steady_tick(std::time::Duration::from_millis(200));
            pb.tick();
            pb
        };

        let p2x = {
            let pb = mb.add(ProgressBar::new(segmentsize - progress * NONCE_SIZE));
            pb.set_style(ProgressStyle::default_bar()
                .template("{prefix:>12} {wide_bar} {bytes:>8} {bytes_per_sec:>10}")
                .expect("Failed to set template")
                .progress_chars("██░"));
            pb.set_prefix("Writing:");
            pb.enable_steady_tick(std::time::Duration::from_millis(200));
            pb.tick();
            pb
        };

        let sw = Stopwatch::start_new();

        let task = Arc::new(task);

        let thread_pinning = true;
        let core_ids = if thread_pinning {
            core_affinity::get_core_ids().unwrap()
        } else {
            Vec::new()
        };

        let hasher = thread::spawn({
            create_scheduler_thread(
                task.clone(),
                rayon::ThreadPoolBuilder::new()
                    .num_threads(task.cpu_threads as usize)
                    .start_handler(move |id| {
                        if thread_pinning {
                            #[cfg(not(windows))]
                            let core_id = core_ids[id % core_ids.len()];
                            #[cfg(not(windows))]
                            core_affinity::set_for_current(core_id);
                            #[cfg(windows)]
                            set_thread_ideal_processor(id % core_ids.len());
                        }
                    })
                    .build()
                    .unwrap(),
                progress,
                Some(p1x),
                rx_empty_buffers.clone(),
                tx_full_buffers.clone(),
                simd_ext,
            )
        });

        let writer = thread::spawn({
            create_writer_thread(
                task.clone(),
                progress,
                Some(p2x),
                rx_full_buffers.clone(),
                tx_empty_buffers.clone(),
            )
        });

        let writer_result = writer.join();
        let hasher_result = hasher.join();

        if let Err(e) = writer_result {
            eprintln!("Writer thread panicked: {:?}", e);
        }
        
        if let Err(e) = hasher_result {
            eprintln!("Hasher thread panicked: {:?}", e);
        }

        let was_stopped = if let Some(stop_flag) = &task.stop_flag {
            stop_flag.load(Ordering::Relaxed)
        } else {
            false
        };

        let _ = mb.clear();

        let elapsed = sw.elapsed_ms() as u64;
        let hours = elapsed / 1000 / 60 / 60;
        let minutes = elapsed / 1000 / 60 - hours * 60;
        let seconds = elapsed / 1000 - hours * 60 * 60 - minutes * 60;

        let completed_nonces = task.nonces - progress;
            
        if was_stopped {

            println!("\nHashing interrupted.");
        } else if completed_nonces > 0 {

            println!(
                "\nGenerated {} nonces in {}h{:02}m{:02}s, {:.2} MiB/s, {:.0} nonces/m.",
                completed_nonces,
                hours,
                minutes,
                seconds,
                completed_nonces as f64 * 1000.0 / (elapsed as f64 + 1.0) / 4.0,
                completed_nonces as f64 * 1000.0 / (elapsed as f64 + 1.0) * 60.0
            );
            println!("Hashing completed!");
        }

        #[cfg(feature = "gui")]
        if let Some(tx) = &task.progress_tx {
            let completed_nonces = task.nonces - progress;
            
            if was_stopped {

                let _ = tx.send(ProgressUpdate::Error("STOP_REQUESTED".to_string()));
            } else if completed_nonces > 0 {

                let _ = tx.send(ProgressUpdate::Log(format!(
                    "\nGenerated {} nonces in {}h{:02}m{:02}s, {:.2} MiB/s, {:.0} nonces/m.",
                    completed_nonces,
                    hours,
                    minutes,
                    seconds,
                    completed_nonces as f64 * 1000.0 / (elapsed as f64 + 1.0) / 4.0,
                    completed_nonces as f64 * 1000.0 / (elapsed as f64 + 1.0) * 60.0
                )));
                let _ = tx.send(ProgressUpdate::Progress(1.0));
                let _ = tx.send(ProgressUpdate::Speed(completed_nonces as f64 * 1000.0 / (elapsed as f64 + 1.0) * 60.0));
                let _ = tx.send(ProgressUpdate::Done);
            }
        }
    }
}

fn calculate_mem_to_use(
    task: &HasherTask,
    memory: &sys_info::MemInfo,
    nonces_per_sector: u64,
    gpu: bool,
    gpu_mem_needed: u64,
) -> Result<u64, &'static str> {
    let segmentsize = task.nonces * NONCE_SIZE;

    let mut mem = match task.mem.parse::<Bytes>() {
        Ok(x) => x.size() as u64,
        Err(_) => {
            println!(
                "Error: Can't parse memory limit parameter, input={}",
                task.mem,
            );
            println!("\nPlease specify a number followed by a unit. If no unit is provided, bytes will be assumed.");
            println!("Supported units: B, KiB, MiB, GiB, TiB, PiB, EiB, KB, MB, GB, TB, PB, EB");
            println!("Example: --mem 10GiB\n");
            println!("Shutting down...");
            return Err("invalid unit");
        }
    };
    
    if gpu && mem > 0 && mem < gpu_mem_needed + nonces_per_sector * NONCE_SIZE {
        println!("Error: Insufficient host memory for GPU hashing!");
        println!("Shutting down...");
        process::exit(0);
    }

    if gpu && mem > 0 {
        mem -= gpu_mem_needed;
    }

    if mem == 0 {
        mem = segmentsize;
    }
    mem = min(mem, segmentsize + gpu_mem_needed);

    let nonces_per_sector = if gpu {
        max(16, nonces_per_sector)
    } else {
        nonces_per_sector
    };

    let avail_mem_bytes = get_avail_mem(&memory) * 1024;
    

    let max_buffer_from_free_mem = (avail_mem_bytes as f64 * 0.75) as u64;
    

    let mem_without_gpu = if gpu && max_buffer_from_free_mem > gpu_mem_needed {
        max_buffer_from_free_mem - gpu_mem_needed
    } else {
        max_buffer_from_free_mem
    };
    
    mem = min(mem, mem_without_gpu);

    let num_buffer = 2;
    mem /= num_buffer * NONCE_SIZE * nonces_per_sector;
    mem *= num_buffer * NONCE_SIZE * nonces_per_sector;

    mem = max(mem, num_buffer * NONCE_SIZE * nonces_per_sector);
    
    println!("Memory calculation:");
    println!("  Available memory: {:.2} GiB", avail_mem_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("  75% of available: {:.2} GiB", max_buffer_from_free_mem as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("  GPU memory needed: {:.2} GiB", gpu_mem_needed as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("  Final buffer size: {:.2} GiB", mem as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("  Buffer configuration: {} buffer(s) of {} nonces each", num_buffer, mem / NONCE_SIZE / num_buffer);
    
    #[cfg(feature = "gui")]
    if let Some(tx) = &task.progress_tx {
        let _ = tx.send(crate::hasher::ProgressUpdate::Log(
            format!("Buffer size: {:.2} GiB ({} buffers)", 
                   mem as f64 / 1024.0 / 1024.0 / 1024.0, num_buffer)
        ));
    }

    Ok(mem)
}

#[cfg(not(windows))]
fn get_avail_mem(memory: &sys_info::MemInfo) -> u64 {
    memory.avail
}

#[cfg(windows)]
fn get_avail_mem(memory: &sys_info::MemInfo) -> u64 {
    memory.free
}