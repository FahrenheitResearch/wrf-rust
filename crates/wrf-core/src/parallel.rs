//! Fork-aware rayon thread pool.
//!
//! rayon's global pool does not survive `fork()`: the child inherits the pool's
//! bookkeeping but none of its worker threads, so the first parallel job it
//! submits waits forever. Python's `ProcessPoolExecutor` forks its workers on
//! Linux, so any process that touched the global pool before forking would
//! hang its workers on their first chunked read or parallel diagnostic.
//!
//! Work run through [`install`] uses a pool owned by the current process and
//! rebuilt whenever the process id changes. Nested rayon calls, including
//! those inside dependencies, run on whichever pool the caller is already in,
//! so wrapping an entry point covers everything beneath it.

use std::sync::{Arc, Mutex};

use rayon::{ThreadPool, ThreadPoolBuilder};

static POOL: Mutex<Option<(u32, Arc<ThreadPool>)>> = Mutex::new(None);

fn current_pool(pid: u32) -> Option<Arc<ThreadPool>> {
    let slot = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    match slot.as_ref() {
        Some((owner, pool)) if *owner == pid => Some(Arc::clone(pool)),
        _ => None,
    }
}

fn pool() -> Arc<ThreadPool> {
    let pid = std::process::id();
    if let Some(pool) = current_pool(pid) {
        return pool;
    }

    // Build outside the lock so a fork can never inherit it held for long.
    // The default thread count honors RAYON_NUM_THREADS like the global pool.
    let built = Arc::new(
        ThreadPoolBuilder::new()
            .thread_name(|index| format!("wrf-rayon-{index}"))
            .build()
            .expect("failed to build wrf-core rayon thread pool"),
    );

    let mut slot = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    match slot.as_ref() {
        // Another thread finished building first; use its pool.
        Some((owner, pool)) if *owner == pid => Arc::clone(pool),
        _ => {
            // A pool inherited across fork() has no threads in this process,
            // so dropping it could wait on them. Leak it instead.
            if let Some((_, stale)) = slot.replace((pid, Arc::clone(&built))) {
                std::mem::forget(stale);
            }
            built
        }
    }
}

/// Run `op` on this process's rayon pool, rebuilding the pool after a fork.
pub fn install<R, OP>(op: OP) -> R
where
    R: Send,
    OP: FnOnce() -> R + Send,
{
    pool().install(op)
}
