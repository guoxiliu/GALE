import argparse

p = argparse.ArgumentParser()
p.add_argument('file', nargs="+", help="STDOUT from GALE TTK ScalarFieldCriticalPoints run")
args = p.parse_args()

suffix100s = ['Triangles and boundary triangles preconditioned in',
              'Time usage for preconditioning:',
              '[ScalarFieldCriticalPoints] Processed',
              'Time usage for computation:',
              '[TIMER]',
              '[TTK:TIMER]',
              ]

averaging = dict()
by_class = dict()

for fname in args.file:
    print(fname)
    with open(fname, 'r') as f:
        lines = f.readlines()
        keep = list()
        keep_idx = list()
        for line_idx, line in enumerate(lines):
            if 's|' in line:
                keep.append(line.rstrip())
                keep_idx.append(line_idx)
            elif 'Memory usage' in line:
                keep.append(line.rstrip())
                keep_idx.append(line_idx)
            elif any([_ in line for _ in suffix100s]):
                keep.append(line.rstrip())
                keep_idx.append(line_idx)

    total_time = 0.0
    for line_idx, line in zip(keep_idx, keep):
        if 'Memory usage' in line:
            continue
        elif 's|' in line:
            try:
                what_idx = line.index('Built')+len('Built ')
            except ValueError:
                continue
            try:
                what_next = what_idx+line[what_idx:].index('.')-1
            except ValueError:
                print("\t", line[what_idx:])
                raise
            what = line[what_idx:what_next]
            cutoff = what_next+line[what_next:].index('s|')
            segment = what_next+line[what_next:cutoff].rindex('[')+1
            section = line[segment:cutoff]
        else:
            for maybe in suffix100s:
                if maybe in line:
                    if maybe != "[TIMER]":
                        what = maybe
                    else:
                        what = line[8:line.rindex(' ')]
                    # Believe it or not, the spacing is inconsistent so I'm just
                    # doing this to account for it; never heard of regex who's that
                    line = line.rstrip()
                    if (maybe == "Time usage for computation:") and ('s' in line):
                        line = line[:line.rindex('s')-1].rstrip()
                    while line[-1] not in "0123456789.e-":
                        line = line[:-1]
                    section = line[line.rindex(' ')+1:]
                    break
        if what == 'VT: [TIMER] VV:':
            raise ValueError(f"Race condition in output detected in {fname}:{line_idx}")
        #print("\t", what, '---', section)
        try:
            total_time += float(section)
            if what not in by_class:
                by_class[what] = list()
            by_class[what].append(float(section))
        except ValueError:
            print("\t", line)

    print(f"Total time for {fname}: {total_time}")
    if 'iter' in fname:
        aname = fname[:fname.index('iter')]
        if aname not in averaging:
            averaging[aname] = list()
        averaging[aname].append(total_time)

for (dname, dictionary) in zip(["Overall", "Breakdown"],[averaging, by_class]):
    for aname, times in dictionary.items():
        print(f"[{dname}] Average time for '{aname}' (x{len(times)}): {sum(times)/len(times)} (Average per file: {sum(times)/len(args.file)})")

