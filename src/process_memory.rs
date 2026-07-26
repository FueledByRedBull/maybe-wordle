#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProcessMemorySnapshot {
    pub current_working_set_bytes: u64,
    pub peak_working_set_bytes: u64,
}

#[cfg(windows)]
pub(crate) fn process_memory_snapshot() -> Option<ProcessMemorySnapshot> {
    use std::{ffi::c_void, mem::size_of};

    #[repr(C)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCountersEx,
            size: u32,
        ) -> i32;
    }

    let mut counters = ProcessMemoryCountersEx {
        cb: size_of::<ProcessMemoryCountersEx>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
        private_usage: 0,
    };
    // SAFETY: both functions are process-local Windows APIs. `counters` is a live,
    // correctly sized writable structure for the duration of the call.
    let succeeded = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            size_of::<ProcessMemoryCountersEx>() as u32,
        )
    } != 0;
    succeeded.then_some(ProcessMemorySnapshot {
        current_working_set_bytes: counters.working_set_size as u64,
        peak_working_set_bytes: counters.peak_working_set_size as u64,
    })
}

#[cfg(target_os = "linux")]
pub(crate) fn process_memory_snapshot() -> Option<ProcessMemorySnapshot> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib = |label: &str| {
        status
            .lines()
            .find(|line| line.starts_with(label))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()
    };
    Some(ProcessMemorySnapshot {
        current_working_set_bytes: kib("VmRSS:")?.saturating_mul(1024),
        peak_working_set_bytes: kib("VmHWM:")?.saturating_mul(1024),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn process_memory_snapshot() -> Option<ProcessMemorySnapshot> {
    use std::{ffi::c_int, mem::size_of};

    type MachPort = u32;
    type MachMessageTypeNumber = u32;

    #[repr(C)]
    struct TimeValue {
        seconds: c_int,
        microseconds: c_int,
    }

    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: TimeValue,
        system_time: TimeValue,
        policy: c_int,
        suspend_count: c_int,
    }

    const KERN_SUCCESS: c_int = 0;
    const MACH_TASK_BASIC_INFO: c_int = 20;

    #[link(name = "System")]
    unsafe extern "C" {
        fn mach_task_self() -> MachPort;
        fn task_info(
            target_task: MachPort,
            flavor: c_int,
            task_info_out: *mut c_int,
            task_info_out_count: *mut MachMessageTypeNumber,
        ) -> c_int;
    }

    let mut info = MachTaskBasicInfo {
        virtual_size: 0,
        resident_size: 0,
        resident_size_max: 0,
        user_time: TimeValue {
            seconds: 0,
            microseconds: 0,
        },
        system_time: TimeValue {
            seconds: 0,
            microseconds: 0,
        },
        policy: 0,
        suspend_count: 0,
    };
    let mut count = (size_of::<MachTaskBasicInfo>() / size_of::<c_int>()) as MachMessageTypeNumber;
    // SAFETY: `mach_task_self` returns the current task send right. `info` is a
    // correctly sized writable `mach_task_basic_info` buffer and `count` reports
    // that buffer in the integer units required by `task_info`.
    let result = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            (&mut info as *mut MachTaskBasicInfo).cast::<c_int>(),
            &mut count,
        )
    };
    (result == KERN_SUCCESS).then_some(ProcessMemorySnapshot {
        current_working_set_bytes: info.resident_size,
        peak_working_set_bytes: info.resident_size_max,
    })
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub(crate) fn process_memory_snapshot() -> Option<ProcessMemorySnapshot> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(windows, target_os = "linux", target_os = "macos"))]
    fn process_memory_is_available_and_sane() {
        let snapshot = process_memory_snapshot().expect("memory snapshot");
        assert!(snapshot.current_working_set_bytes > 0);
        assert!(snapshot.peak_working_set_bytes >= snapshot.current_working_set_bytes);
    }
}
