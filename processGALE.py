import argparse
import pathlib
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument('file', nargs="*", help="STDOUT from GALE criticalPointsKernel run (or stdin if none specified)")
args = p.parse_args()
if len(args.file) == 0:
    args.file.append(None)
accepted = list()
dropped = list()
for fname in args.file:
    if pathlib.Path(fname).exists():
        accepted.append(fname)
    else:
        dropped.append(fname)
args.file = accepted
if len(dropped) > 0:
    print(f"Dropped {len(dropped)} filenames that could not be found:")
    print("\t"+"\n\t".join(dropped))

"""
[TIMER] Argument parsing: 0.000114877
[TIMER] Create CUDA Context: 0.255254
[TIMER] Initialize VTK: 0.0545443
[TIMER] GALE VV: [CPU 6] 0.0140805
[TIMER] GALE SFCP preparation [CPU 6]: 0.00166462
[ScalarFieldCriticalPoints] Time usage for computation [CPU 6]: 0.000203672 s.
[TIMER] Retrieve results from GPU [CPU 6]: 0.000141696
[TIMER] Tabulate partition results [CPU 6]: 3.6976e-05
[TIMER] MAIN: 0.0134
"""

search_strs = ["[TIMER] Main:",
               "[TIMER] Preprocessing:",
               "[TIMER] Algorithm:",
               #"[ScalarFieldCriticalPoints] Time usage for computation",
               ]

# Key (entry of search_strs) | Value (substring that lets you know to exclude it)
exclude = defaultdict(list)

averaging = dict()
breakdown = dict()
for fname in args.file:
    if fname is None:
        lines = sys.stdin.readlines()
        fname = 'stdin'
    else:
        with open(fname, 'r') as f:
            lines = f.readlines()
    print(fname)
    keep = list()
    for line in lines:
        for trigger in search_strs:
            if trigger not in line:
                continue
            excluded = False
            for exclusion in exclude[trigger]:
                if exclusion in line:
                    excluded = True
                    break
            if not excluded:
                keep.append(line.rstrip())
    if 'iter' in fname:
        aname = fname[:fname.index('iter')]
    else:
        aname = fname
    if aname not in averaging:
        averaging[aname] = list()
        breakdown[aname] = dict()

    total_time = 0.0
    for line in keep:
        for trigger in search_strs:
            if trigger in line:
                what = trigger
                section = line.rstrip(' s.')
                section = section[section.rindex(' ')+1:]
                break
        print("\t"+f"{what} ({line}) --- {section}")
        try:
            breakdown_time = float(section)
            if what not in breakdown[aname]:
                breakdown[aname][what] = list()
            breakdown[aname][what].append(breakdown_time)
            total_time += breakdown_time
        except ValueError:
            print("\t", line)

    print(f"Total time for {fname}: {total_time}")
    averaging[aname].append(total_time)

for aname, times in averaging.items():
    print(f"Average time for {aname}: {sum(times)/len(times):.3f}")
    for what in breakdown[aname].keys():
        print("\t"+f"{what}: {sum(breakdown[aname][what])/len(breakdown[aname][what]):.3f}")

