import pathlib
import tempfile
import unittest

from lexbrowser_eval.lexbench.qwen3_8b.stress_monitor import (
    RESOURCE_CSV_FIELDS,
    calculate_cpu_cores,
    calculate_throughput_per_hour,
    read_cgroup_processes,
    read_cgroup_stats,
    read_host_memory,
    summarize_processes,
)


class CgroupStatsTests(unittest.TestCase):
    def test_reads_cpu_memory_oom_and_pid_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "cpu.stat").write_text(
                "usage_usec 9000000\nuser_usec 7000000\nsystem_usec 2000000\n",
                encoding="utf-8",
            )
            (root / "memory.current").write_text("1048576\n", encoding="utf-8")
            (root / "memory.peak").write_text("2097152\n", encoding="utf-8")
            (root / "memory.high").write_text("3145728\n", encoding="utf-8")
            (root / "memory.max").write_text("max\n", encoding="utf-8")
            (root / "memory.events").write_text(
                "low 0\nhigh 4\nmax 1\noom 2\noom_kill 1\noom_group_kill 0\n",
                encoding="utf-8",
            )
            (root / "pids.current").write_text("17\n", encoding="utf-8")
            (root / "pids.peak").write_text("23\n", encoding="utf-8")
            (root / "pids.max").write_text("8192\n", encoding="utf-8")

            stats = read_cgroup_stats(root)

        self.assertEqual(stats.cpu_usage_usec, 9_000_000)
        self.assertEqual(stats.cpu_user_usec, 7_000_000)
        self.assertEqual(stats.cpu_system_usec, 2_000_000)
        self.assertEqual(stats.memory_current_bytes, 1_048_576)
        self.assertEqual(stats.memory_peak_bytes, 2_097_152)
        self.assertEqual(stats.memory_high_bytes, 3_145_728)
        self.assertIsNone(stats.memory_max_bytes)
        self.assertEqual(stats.memory_events_high, 4)
        self.assertEqual(stats.memory_events_oom, 2)
        self.assertEqual(stats.memory_events_oom_kill, 1)
        self.assertEqual(stats.pids_current, 17)
        self.assertEqual(stats.pids_peak, 23)
        self.assertEqual(stats.pids_max, 8192)


class ProcessAttributionTests(unittest.TestCase):
    def _write_process(
        self,
        proc_root: pathlib.Path,
        pid: int,
        ppid: int,
        name: str,
        command: list[str],
        pss_kib: int,
        rss_kib: int,
    ) -> None:
        process_dir = proc_root / str(pid)
        process_dir.mkdir()
        (process_dir / "status").write_text(f"Name:\t{name}\nPPid:\t{ppid}\n", encoding="utf-8")
        (process_dir / "comm").write_text(f"{name}\n", encoding="utf-8")
        (process_dir / "cmdline").write_bytes(b"\0".join(part.encode() for part in command) + b"\0")
        (process_dir / "smaps_rollup").write_text(
            f"Rss: {rss_kib} kB\nPss: {pss_kib} kB\n", encoding="utf-8"
        )

    def test_attributes_agent_and_chrome_pss_inside_the_cgroup_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cgroup = root / "cgroup"
            proc = root / "proc"
            cgroup.mkdir()
            proc.mkdir()
            (cgroup / "cgroup.procs").write_text("101\n102\n103\n104\n", encoding="utf-8")
            self._write_process(
                proc,
                101,
                1,
                "python",
                [
                    "python",
                    "browseruse_bench/runner/agent_runner.py",
                    "--workspace",
                    "/run/tasks/41",
                ],
                100,
                120,
            )
            self._write_process(
                proc,
                102,
                101,
                "chrome",
                ["/opt/google/chrome/chrome", "--remote-debugging-port=1"],
                200,
                250,
            )
            self._write_process(
                proc,
                103,
                102,
                "chrome",
                ["/opt/google/chrome/chrome", "--type=renderer"],
                300,
                350,
            )
            self._write_process(proc, 104, 1, "uv", ["uv", "run", "bubench"], 50, 60)
            # An unrelated host Chrome exists under /proc but not cgroup.procs.
            self._write_process(
                proc,
                999,
                1,
                "chrome",
                ["/opt/google/chrome/chrome"],
                5000,
                6000,
            )

            processes = read_cgroup_processes(cgroup, proc_root=proc)
            stats = summarize_processes(processes)

        self.assertEqual(stats.process_count, 4)
        self.assertEqual(stats.active_agent_runners, 1)
        self.assertEqual(stats.chrome_process_count, 2)
        self.assertEqual(stats.chrome_session_count, 1)
        self.assertEqual(stats.process_tree_pss_bytes, 650 * 1024)
        self.assertEqual(stats.chrome_pss_bytes, 500 * 1024)
        self.assertEqual(stats.nonchrome_pss_bytes, 150 * 1024)


class GuardAndMetricTests(unittest.TestCase):
    def test_reads_host_memory_guard_values_from_proc_meminfo(self):
        with tempfile.TemporaryDirectory() as tmp:
            meminfo = pathlib.Path(tmp) / "meminfo"
            meminfo.write_text(
                "MemTotal:       131677308 kB\n"
                "MemAvailable:    85110640 kB\n"
                "SwapTotal:        2097148 kB\n"
                "SwapFree:          155928 kB\n",
                encoding="utf-8",
            )

            memory = read_host_memory(meminfo)

        self.assertEqual(memory.total_bytes, 131_677_308 * 1024)
        self.assertEqual(memory.available_bytes, 85_110_640 * 1024)
        self.assertEqual(memory.swap_total_bytes, 2_097_148 * 1024)
        self.assertEqual(memory.swap_free_bytes, 155_928 * 1024)
        self.assertTrue(memory.below_available_reserve(86_000_000 * 1024))
        self.assertFalse(memory.below_available_reserve(32 * 1024**3))

    def test_calculates_attributed_cpu_and_completion_throughput(self):
        self.assertAlmostEqual(calculate_cpu_cores(1_000_000, 9_000_000, 2.0), 4.0)
        self.assertAlmostEqual(calculate_throughput_per_hour(20, 600.0), 120.0)
        with self.assertRaises(ValueError):
            calculate_cpu_cores(9_000_000, 1_000_000, 2.0)
        with self.assertRaises(ValueError):
            calculate_throughput_per_hour(20, 0.0)

    def test_resource_csv_contract_contains_attributed_and_guard_fields(self):
        required = {
            "cpu_usage_usec",
            "cpu_delta_usec",
            "cpu_cores",
            "memory_current_bytes",
            "memory_events_oom_kill",
            "pids_current",
            "agent_runner_active",
            "chrome_session_active",
            "process_tree_pss_bytes",
            "chrome_pss_bytes",
            "terminal_instance_count",
            "throughput_60s_task_per_hour",
            "host_memory_available_bytes",
            "host_swap_free_bytes",
        }

        self.assertTrue(required.issubset(RESOURCE_CSV_FIELDS))
        self.assertNotIn("host_memory_used_bytes", RESOURCE_CSV_FIELDS)


if __name__ == "__main__":
    unittest.main()
