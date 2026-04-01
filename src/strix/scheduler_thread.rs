//! Dedicated scheduler thread (STRIX Protocol §7, §9.2 — Thread B).
//!
//! `SchedulerThread` spawns a real `std::thread` that runs the
//! residency scheduler on a configurable tick interval (default 2 ms).
//!
//! Hardware/OS agnostic — uses only `std::thread` and `std::sync`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// ── SchedulerStats ──────────────────────────────────────────────────────

/// Runtime statistics collected by the scheduler thread.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Total ticks executed since the thread started.
    pub ticks: u64,
    /// Timestamp when the thread was started.
    pub started_at: Instant,
    /// Average tick duration in microseconds (0 if no ticks yet).
    pub avg_tick_us: u64,
}

// ── Shared state ────────────────────────────────────────────────────────

/// Shared state between the scheduler thread and the owning code.
struct SharedState {
    /// Shutdown signal — when `true`, the thread exits.
    shutdown: AtomicBool,
    /// Tick counter.
    tick_count: AtomicU64,
    /// Cumulative tick duration in microseconds (for averaging).
    cumulative_tick_us: AtomicU64,
}

// ── Tick callback ───────────────────────────────────────────────────────

/// The work executed on each scheduler tick.
///
/// Implementors drive the `ResidencyScheduler::tick()` cycle, VRAM
/// pressure evaluation, and I/O completion polling — whatever the
/// scheduler needs to do every 2 ms.
pub trait SchedulerWork: Send + 'static {
    /// Called once per tick. Should be fast (< 2 ms target).
    fn tick(&mut self);
}

/// Blanket implementation: any `FnMut()` is valid scheduler work.
impl<F: FnMut() + Send + 'static> SchedulerWork for F {
    fn tick(&mut self) {
        self()
    }
}

// ── SchedulerThread ─────────────────────────────────────────────────────

/// A dedicated thread that runs scheduler ticks at a fixed interval.
///
/// # Lifecycle
///
/// 1. `spawn(work, interval)` — starts the thread.
/// 2. The thread loops: `work.tick()`, then `sleep(interval)`.
/// 3. `shutdown()` — signals the thread to stop and joins it.
/// 4. Dropping `SchedulerThread` also triggers shutdown.
pub struct SchedulerThread {
    handle: Option<JoinHandle<()>>,
    state: Arc<SharedState>,
    started_at: Instant,
}

impl SchedulerThread {
    /// Spawn the scheduler thread.
    ///
    /// - `work`: the object whose `tick()` method is called each cycle.
    /// - `interval`: sleep duration between ticks (e.g. `Duration::from_millis(2)`).
    pub fn spawn(mut work: impl SchedulerWork, interval: Duration) -> Self {
        let state = Arc::new(SharedState {
            shutdown: AtomicBool::new(false),
            tick_count: AtomicU64::new(0),
            cumulative_tick_us: AtomicU64::new(0),
        });
        let thread_state = Arc::clone(&state);
        let started_at = Instant::now();

        let handle = thread::Builder::new()
            .name("strix-scheduler".into())
            .spawn(move || {
                while !thread_state.shutdown.load(Ordering::Relaxed) {
                    let tick_start = Instant::now();
                    work.tick();
                    let elapsed_us = tick_start.elapsed().as_micros() as u64;

                    thread_state.tick_count.fetch_add(1, Ordering::Relaxed);
                    thread_state
                        .cumulative_tick_us
                        .fetch_add(elapsed_us, Ordering::Relaxed);

                    // Sleep the remaining interval (if tick was faster than interval).
                    let tick_dur = tick_start.elapsed();
                    if tick_dur < interval {
                        thread::sleep(interval - tick_dur);
                    }
                }
            })
            .expect("failed to spawn strix-scheduler thread");

        Self {
            handle: Some(handle),
            state,
            started_at,
        }
    }

    /// Request the thread to shut down and wait for it to finish.
    ///
    /// This is idempotent — calling it multiple times is safe.
    pub fn shutdown(&mut self) {
        self.state.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Whether a shutdown has been requested.
    pub fn is_shutdown_requested(&self) -> bool {
        self.state.shutdown.load(Ordering::Relaxed)
    }

    /// Whether the thread has finished (joined).
    pub fn is_finished(&self) -> bool {
        self.handle.is_none()
    }

    /// Current statistics snapshot.
    pub fn stats(&self) -> SchedulerStats {
        let ticks = self.state.tick_count.load(Ordering::Relaxed);
        let cumulative_us = self.state.cumulative_tick_us.load(Ordering::Relaxed);
        let avg_tick_us = if ticks > 0 {
            cumulative_us / ticks
        } else {
            0
        };
        SchedulerStats {
            ticks,
            started_at: self.started_at,
            avg_tick_us,
        }
    }
}

impl Drop for SchedulerThread {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn spawn_tick_and_shutdown() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut thread = SchedulerThread::spawn(
            move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            },
            Duration::from_millis(1),
        );

        // Let it tick a few times.
        thread::sleep(Duration::from_millis(20));
        thread.shutdown();

        let ticks = counter.load(Ordering::SeqCst);
        assert!(ticks >= 5, "expected ≥5 ticks in 20ms, got {ticks}");

        let stats = thread.stats();
        assert_eq!(stats.ticks, ticks as u64);
        assert!(thread.is_finished());
    }

    #[test]
    fn double_shutdown_safe() {
        let mut thread = SchedulerThread::spawn(|| {}, Duration::from_millis(10));
        thread.shutdown();
        thread.shutdown(); // should not panic
        assert!(thread.is_finished());
    }

    #[test]
    fn stats_tick_count() {
        let mut thread = SchedulerThread::spawn(|| {}, Duration::from_millis(1));
        thread::sleep(Duration::from_millis(15));
        let stats = thread.stats();
        assert!(stats.ticks >= 3, "expected ≥3 ticks, got {}", stats.ticks);
        thread.shutdown();
    }

    #[test]
    fn drop_triggers_shutdown() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        {
            let _thread = SchedulerThread::spawn(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                },
                Duration::from_millis(1),
            );
            thread::sleep(Duration::from_millis(10));
        } // Drop here → shutdown
        let final_count = counter.load(Ordering::SeqCst);
        // After drop, counter should stop incrementing.
        thread::sleep(Duration::from_millis(10));
        let after_drop = counter.load(Ordering::SeqCst);
        assert_eq!(final_count, after_drop, "thread must stop on Drop");
    }

    #[test]
    fn scheduler_work_trait_with_struct() {
        struct Counter {
            count: u32,
        }
        impl SchedulerWork for Counter {
            fn tick(&mut self) {
                self.count += 1;
            }
        }
        let mut thread = SchedulerThread::spawn(Counter { count: 0 }, Duration::from_millis(1));
        thread::sleep(Duration::from_millis(10));
        assert!(thread.stats().ticks >= 3);
        thread.shutdown();
    }
}
