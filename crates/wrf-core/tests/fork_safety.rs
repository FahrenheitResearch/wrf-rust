//! Rayon work must not hang in a process forked after the parent already used
//! the thread pool. This is the Python `ProcessPoolExecutor` pattern on Linux,
//! where worker processes are created with `fork()`.
#![cfg(unix)]

use std::time::{Duration, Instant};

use rayon::prelude::*;

const EXPECTED_SUM: u64 = 99_990_000;

fn parallel_sum() -> u64 {
    // Enough items that rayon splits the work and hands it to pool threads.
    wrf_core::parallel::install(|| (0..10_000u64).into_par_iter().map(|v| v * 2).sum())
}

#[test]
fn install_completes_in_forked_child() {
    // Start the pool in the parent before forking.
    assert_eq!(parallel_sum(), EXPECTED_SUM);

    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork failed");
    if pid == 0 {
        let code = if parallel_sum() == EXPECTED_SUM { 0 } else { 1 };
        unsafe { libc::_exit(code) };
    }

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut status = 0;
    loop {
        let reaped = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
        if reaped == pid {
            break;
        }
        if Instant::now() > deadline {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
                libc::waitpid(pid, &mut status, 0);
            }
            panic!("forked child hung running parallel work");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
        "forked child failed, wait status {status}"
    );
}
