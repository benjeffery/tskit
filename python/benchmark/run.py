import json
import math
import os.path
import platform
import subprocess
import sys
import tempfile
import timeit
from pathlib import Path
import click

import psutil
import tqdm
import yaml
from matplotlib.colors import LinearSegmentedColormap
from si_prefix import si_format

tskit_dir = Path(__file__).parent.parent
sys.path.append(str(tskit_dir))
import tskit  # noqa: E402
import msprime  # noqa: E402

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def system_info():
    ret = {}
    uname = platform.uname()
    for attr in ["system", "node", "release", "version", "machine", "processor"]:
        ret[attr] = getattr(uname, attr)
    ret["python_version"] = sys.version
    cpufreq = psutil.cpu_freq()
    ret["physical_cores"] = psutil.cpu_count(logical=False)
    ret["total_cores"] = psutil.cpu_count(logical=True)
    ret["max_frequency"] = cpufreq.max
    ret["min_frequency"] = cpufreq.min
    ret["current_frequency"] = cpufreq.current
    ret["cpu_usage_per_core"] = [
        percentage for percentage in psutil.cpu_percent(percpu=True, interval=1)
    ]
    ret["total_cpu_usage"] = psutil.cpu_percent()
    return ret


def make_file():
    benchmark_trees = tskit_dir / "benchmark" / "bench.trees"
    if not os.path.exists(benchmark_trees):
        print("Generating benchmark trees...")
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=10_000)
        demography.add_population(name="B", initial_size=5_000)
        demography.add_population(name="C", initial_size=1_000)
        demography.add_population_split(time=1000, derived=["A", "B"], ancestral="C")
        ts = msprime.sim_ancestry(
            samples={"A": 25000, "B": 25000},
            demography=demography,
            sequence_length=1_000_000,
            random_seed=42,
            recombination_rate=0.0000001,
            record_migrations=True,
            record_provenance=True,
        )
        ts = msprime.sim_mutations(ts, rate=0.000001, random_seed=42)
        ts.dump(benchmark_trees)
        ts = msprime.sim_ancestry(
            samples={"A": 1, "B": 1},
            demography=demography,
            sequence_length=1,
            random_seed=42,
            recombination_rate=0,
            record_migrations=True,
            record_provenance=True,
        )
        ts = msprime.sim_mutations(ts, rate=0.001, random_seed=42)
        ts.dump(tskit_dir / "benchmark" / "tiny.trees")


def autotime(setup, code, quick):
    t = timeit.Timer(setup=setup, stmt=code)
    try:
        one_run = t.timeit(number=1)
    except Exception as e:
        print(f"{code}: Error running benchmark: {e}")
        return None
    num_trials = int(max(1, (.2 if quick else 2) / one_run))
    return one_run, num_trials, t.timeit(number=num_trials) / num_trials

def benchmark_memory(setup, code):
    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = os.path.join(tmpdir, "tmp.py")
        with open(code_file, "w") as f:
            f.write(f"{setup}\n{code}")
#        f"valgrind --tool=massif --pages-as-heap=yes --massif-out-file=massif.out "
#        f"python {code_file};
        proc = subprocess.Popen(
            ["valgrind", "--tool=massif", "--pages-as-heap=yes", "--massif-out-file=massif.out", "python", code_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"{code}: Error running benchmark: {stderr}")
            return None
        with open("massif.out") as f:
            max_mem = 0
            for line in f:
                if line.startswith("mem_heap_B="):
                    mem = int(line.split("=")[1])
                    if mem > max_mem:
                        max_mem = mem
        if max_mem == 0:
            print(f"{code}: Error running benchmark: no results from valgrind")
            return None
        return max_mem


def run_benchmarks(cpu, mem, quick):
    results = {}
    setup_memory_results = {}
    for benchmark in tqdm.tqdm(config["benchmarks"]):
        bench_name = benchmark.get("name", benchmark["code"])
        params = benchmark.get("parameters", {"noop": [None]})

        # Expand the parameters
        def sub_expand(context, name, d):
            if isinstance(d, dict):
                ret = []
                for k, v in d.items():
                    new_context = {**{k: v for k, v in context.items()}, name: k}
                    for k2, v2 in v.items():
                        ret += sub_expand(new_context, k2, v2)
                return ret
            elif isinstance(d, list):
                return [
                    {**{k: v for k, v in context.items()}, name: value} for value in d
                ]
            else:
                raise ValueError(f"Invalid parameter type: {type(d)}-{d}")

        expanded_params = []
        for k, v in params.items():
            expanded_params += sub_expand({}, k, v)

        for values in expanded_params:
            setup = (
                f"import sys;sys.path.append('{tskit_dir}');"
                + config["setup"].replace("\n", "\n")
                + benchmark.get("setup", "").replace("\n", "\n").format(**values)
            )
            code = benchmark["code"].replace("\n", "\n").format(**values)
            results.setdefault(bench_name, {})[code] = {}
            if cpu:
                result = autotime(setup, code, quick)
                if result is not None:
                    one_run, num_trials, avg = result
                    results[bench_name][code]["one_run"] = one_run
                    results[bench_name][code]["num_trials"] = num_trials
                    results[bench_name][code]["avg"] = avg
            if mem:
                #First measure the memory usage of the setup
                try:
                    setup_memory = setup_memory_results[setup]
                except KeyError:
                    setup_memory = benchmark_memory(setup, "pass")
                    setup_memory_results[setup] = setup_memory
                memory = benchmark_memory(setup, code)
                if setup_memory is not None and memory is not None:
                    results[bench_name][code]["mem"] = memory - setup_memory
    return results


def generate_report(all_versions_results):
    all_benchmarks = {}
    for _version, results in all_versions_results.items():
        for benchmark, values in results["tskit_benchmarks"].items():
            for code in values.keys():
                all_benchmarks.setdefault(benchmark, set()).add(code)

    all_versions = sorted(all_versions_results.keys())

    cmap = LinearSegmentedColormap.from_list("rg", ["g", "w", "r"], N=256)

    with open(tskit_dir / "benchmark" / "bench-results.html", "w") as f:
        f.write("<html><body>\n")
        f.write("<h1>tskit benchmark results</h1>\n")
        for name, key in [("CPU", "avg"), ("MEM", "mem")]:
            f.write(f"<h2>{name}</h2>\n")
            f.write("<table>\n")
            f.write("<tr><th></th>")
            for version in all_versions:
                f.write(f"<th>{version}</th>")
            f.write("</tr>\n")
            for benchmark in sorted(all_benchmarks.keys()):
                values = all_benchmarks[benchmark]
                indent = False
                if len(values) > 1:
                    indent = True
                    f.write(
                        f"<tr>"
                        f"  <td style='font-family: monospace'>"
                        f"    {benchmark}"
                        f"  </td>"
                        f"</tr>\n"
                    )
                for code in sorted(values):
                    f.write(
                        f"<tr><td style='font-family: monospace;"
                        f"padding-left: {'10px' if indent else 'inherit'}'>{code}</td>"
                    )
                    last_avg = None
                    for version in all_versions:
                        try:
                            avg = all_versions_results[version]["tskit_benchmarks"] \
                                [benchmark][code][key]
                            if last_avg is not None:
                                if last_avg != 0:
                                    percent_change = 100 * ((avg - last_avg) / last_avg)
                                    col = cmap(int(((percent_change / 100) * 128) + 128))
                                else:
                                    percent_change = None
                                    col = (1, 1, 1)
                                # There is some jitter in the memory results so
                                # don't colour if we're below a small change
                                if key == "mem" and math.fabs(avg-last_avg) < 9000:
                                    percent_change = None
                                    col = (1, 1, 1)
                                #For memory benchmarks we get jitter
                                f.write(
                                    f"<td style='background-color: rgba({col[0]*255},"
                                    f" {col[1]*255}, {col[2]*255}, 1)'>"
                                )

                                f.write(f"{si_format(avg)} ")
                                if percent_change is not None:
                                    f.write(f"({percent_change:.1f}%)")
                            else:
                                f.write(f"<td>{si_format(avg)}</td>")
                            last_avg = avg
                        except KeyError:
                            f.write("<td>N/A</td>")

                    f.write("</tr>\n")
            f.write("</table>\n")


import click

@click.command()
@click.option('--no-cpu', default=False, is_flag=True, help="Don't run cpu benchmarks")
@click.option('--mem', default=False, is_flag=True, help='Run memory benchmarks')
@click.option('--quick', default=False, is_flag=True, help='Run quick cpu benchmarks that are less accurate')
@click.option('--report-only', default=False, is_flag=True, help='Only generate the HTML report')

def benchmark(no_cpu, mem, quick, report_only):
    all_versions_results = {}
    results_json = tskit_dir / "benchmark" / "bench-results.json"
    if os.path.exists(results_json):
        with open(results_json) as f:
            all_versions_results = json.load(f)

    if not report_only:
        print("Benchmarking tskit version:", tskit._version.tskit_version)
        make_file()
        results = {}
        results["system"] = system_info()
        results["tskit_benchmarks"] = run_benchmarks(not no_cpu, mem, quick)
        all_versions_results[tskit._version.tskit_version] = results
        with open(results_json, "w") as f:
            json.dump(all_versions_results, f, indent=2)

    generate_report(all_versions_results)
    sys.exit(0)
if __name__ == '__main__':
    benchmark()
