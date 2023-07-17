import pandas as pd

year = 2019
file60 = f"data_analysis/data/{year}_data_1min.hdf5"
file = f"data_analysis/data/{year}_data_10s.hdf5"

exclude = {13, 15, 26, 33}  # PV
if year in {2019, 2020}:
    exclude |= {24}  # no availability
if year == 2020:
    exclude |= {25}  # no availability

mdata = []
psum, ssum, hsum, p60sum, h60sum = None, None, None, None, None

# for i in range(3, 41):
for i in range(3, 42):
    if i in exclude:
        continue
    if i <= 40:
        p = pd.read_hdf(file, f"NO_PV/SFH{i}/HOUSEHOLD/table")["P_TOT"]
        h = pd.read_hdf(file, f"NO_PV/SFH{i}/HEATPUMP/table")["P_TOT"]
        p60 = pd.read_hdf(file60, f"NO_PV/SFH{i}/HOUSEHOLD/table")["P_TOT"]
        h60 = pd.read_hdf(file60, f"NO_PV/SFH{i}/HEATPUMP/table")["P_TOT"]
        s = p + h
        if psum is None:
            psum, ssum, hsum, p60sum, h60sum = p, s, h, p60, h60
        else:
            psum += p.fillna(0)
            ssum += s.fillna(0)
            hsum += h.fillna(0)
            p60sum += p60.fillna(0)
            h60sum += h60.fillna(0)
    else:  # sum
        p, s, h, p60, h60 = psum, ssum, hsum, p60sum, h60sum

    mdata.append({
        "index": i,
        "samples": s.count(),
        "availability": s.count() / len(s),
        "hhAv": p.mean(),
        "hhMin": p.min(),
        "hhMin60": p60.min(),
        "hhMax": p.max(),
        "hhMax60": p60.max(),
        "hhSkew": p.skew(),
        "hpAv": h.mean(),
        "hpMin": h.min(),
        "hpMin60": h60.min(),
        "hpMax": h.max(),
        "hpMax60": h60.max(),
        "hpSkew": h.skew(),
        "hpShare": h.mean() / s.mean(),
        "totAv": s.mean(),
        "totMax": s.max(),
        "totMin": s.min(),
        "totStd": s.std(),
        "totVar": s.var(),
        "totSkew": s.skew(),
        "totKurt": s.kurtosis(),
    })

dm = pd.DataFrame(mdata)
print("availability<75%: ", [dm["index"][i] for i in dm.index if dm["availability"][i] < 0.75])
print("availability<90%: ", [dm["index"][i] for i in dm.index if dm["availability"][i] < 0.90])
print("availability<99%: ", [dm["index"][i] for i in dm.index if dm["availability"][i] < 0.99])
