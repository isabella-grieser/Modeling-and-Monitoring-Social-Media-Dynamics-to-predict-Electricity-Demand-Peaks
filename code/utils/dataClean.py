import pandas as pd



def clean_household_data(path, save):
    df = pd.read_csv(path)

    has_pv = [3, 4, 6]
    no_pv = [1, 2, 5]
    remained_cols = ['utc_timestamp', 'cet_cest_timestamp']
    for i in has_pv:
        df[f"total_power_res_{i}"] = df[f"DE_KN_residential{i}_pv"] + df[
            f"DE_KN_residential{i}_grid_import"] - df[f"DE_KN_residential{i}_grid_export"]
        df[f"total_usage_{i}"] = df[f"total_power_res_{i}"].diff()
        # remained_cols.append(f"DE_KN_residential{i}_pv")
        # remained_cols.append(f"DE_KN_residential{i}_grid_import")
        # remained_cols.append(f"DE_KN_residential{i}_grid_export")
        remained_cols.append(f"total_usage_{i}")
    for i in no_pv:
        df[f"total_power_res_{i}"] = df[f"DE_KN_residential{i}_grid_import"]
        df[f"total_usage_{i}"] = df[f"total_power_res_{i}"].diff()
        # remained_cols.append(f"DE_KN_residential{i}_grid_import")
        remained_cols.append(f"total_usage_{i}")

    df = df.loc[:, df.columns.intersection(remained_cols)]

    df = df.dropna(thresh=2)
    df = df.round(5)
    df.to_csv(save)

def clean_time_series_data(path, save):
    df = pd.read_csv(path)

    remained_cols = ['utc_timestamp', 'cet_cest_timestamp', 'DE_load_actual_entsoe_transparency']

    df = df.loc[:, df.columns.intersection(remained_cols)]
    df = df.dropna()
    df.to_csv(save)


if __name__ == "__main__":

    # clean_household_data("../data/household_data_1min_opsd.csv", "../data/opsd_1min.csv")
    clean_time_series_data("../data/time_series_15min_opsd.csv", "../data/time_15min.csv")