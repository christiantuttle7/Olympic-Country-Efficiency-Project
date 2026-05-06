import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("Datasets/athlete_events.csv").copy()

# Keep Summer Olympics only
df = df[df["Season"] == "Summer"].copy()

# =========================
# 2. BASIC CLEANING
# =========================
df["Gold"] = (df["Medal"] == "Gold").astype(int)
df["Silver"] = (df["Medal"] == "Silver").astype(int)
df["Bronze"] = (df["Medal"] == "Bronze").astype(int)
df["AnyMedal"] = df["Medal"].notna().astype(int)

df["Female"] = (df["Sex"] == "F").astype(int)

# =========================
# 3. HOST COUNTRY MAPPING
# =========================
host_map = {
    1896: "GRE", 1900: "FRA", 1904: "USA", 1906: "GRE",
    1908: "GBR", 1912: "SWE", 1920: "BEL", 1924: "FRA",
    1928: "NED", 1932: "USA", 1936: "GER", 1948: "GBR",
    1952: "FIN", 1956: "AUS", 1960: "ITA", 1964: "JPN",
    1968: "MEX", 1972: "GER", 1976: "CAN", 1980: "URS",
    1984: "USA", 1988: "KOR", 1992: "ESP", 1996: "USA",
    2000: "AUS", 2004: "GRE", 2008: "CHN", 2012: "GBR",
    2016: "BRA"
}

df["HostNOC"] = df["Year"].map(host_map)
df["host_country"] = (df["NOC"] == df["HostNOC"]).astype(int)

# =========================
# 4. AGGREGATE TO COUNTRY-YEAR
# =========================
athletes_sent = (
    df.groupby(["NOC", "Year"])["ID"]
    .nunique()
    .reset_index(name="athletes_sent")
)

sports_participated = (
    df.groupby(["NOC", "Year"])["Sport"]
    .nunique()
    .reset_index(name="sports_participated")
)

events_participated = (
    df.groupby(["NOC", "Year"])["Event"]
    .nunique()
    .reset_index(name="events_participated")
)

athlete_sex = (
    df.groupby(["NOC", "Year", "ID"])["Female"]
    .max()
    .reset_index()
)

female_pct = (
    athlete_sex.groupby(["NOC", "Year"])["Female"]
    .mean()
    .mul(100)
    .reset_index(name="female_athlete_percentage")
)

# =========================
# FIXED MEDAL COUNTS
# Count one medal per country-event, not one medal per athlete
# =========================

medal_rows = df[df["Medal"].notna()].copy()

# One country gets one medal per event per medal type
unique_medals = medal_rows.drop_duplicates(
    subset=["NOC", "Year", "Event", "Medal"]
)

medal_counts = (
    unique_medals.groupby(["NOC", "Year", "Medal"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Make sure all medal columns exist
for medal in ["Gold", "Silver", "Bronze"]:
    if medal not in medal_counts.columns:
        medal_counts[medal] = 0

medal_counts["total_medals"] = (
    medal_counts["Gold"]
    + medal_counts["Silver"]
    + medal_counts["Bronze"]
)

medal_counts = medal_counts[
    ["NOC", "Year", "Gold", "Silver", "Bronze", "total_medals"]
]

host_flag = (
    df.groupby(["NOC", "Year"])["host_country"]
    .max()
    .reset_index()
)

country_year = athletes_sent.merge(sports_participated, on=["NOC", "Year"], how="outer")
country_year = country_year.merge(events_participated, on=["NOC", "Year"], how="outer")
country_year = country_year.merge(female_pct, on=["NOC", "Year"], how="outer")
country_year = country_year.merge(medal_counts, on=["NOC", "Year"], how="outer")
country_year = country_year.merge(host_flag, on=["NOC", "Year"], how="outer")

for col in ["Gold", "Silver", "Bronze", "total_medals", "host_country"]:
    country_year[col] = country_year[col].fillna(0)

country_year["medals_per_athlete"] = (
    country_year["total_medals"] / country_year["athletes_sent"]
)

#Previous Olympic Features
country_year = country_year.sort_values(["NOC", "Year"]).reset_index(drop=True)

country_year["prev_total_medals"] = (
    country_year.groupby("NOC")["total_medals"].shift(1).fillna(0)
)

country_year["prev_medals_per_athlete"] = (
    country_year.groupby("NOC")["medals_per_athlete"].shift(1).fillna(0)
)

# Merge data
country_year["ISO3"] = country_year["NOC"]

pop_df = pd.read_csv("Datasets/population.csv")
pop_df = pop_df[["Country Code", "Year", "Value"]].rename(
    columns={"Country Code": "ISO3", "Value": "population"}
)

gdp_df = pd.read_csv("Datasets/gdp.csv")
gdp_df = gdp_df[["Country Code", "Year", "Value"]].rename(
    columns={"Country Code": "ISO3", "Value": "gdp_total"}
)


country_year = country_year.merge(pop_df, on=["ISO3", "Year"], how="left")
country_year = country_year.merge(gdp_df, on=["ISO3", "Year"], how="left")

country_year["gdp_per_capita"] = (
    country_year["gdp_total"] / country_year["population"]
)

country_year = country_year.drop(columns=["gdp_total"])
country_year["income_group"] = np.nan

#Finalize the new df
final_cols = [
    "NOC", "ISO3", "Year", "population", "gdp_per_capita",
    "income_group", "host_country", "athletes_sent",
    "sports_participated", "events_participated",
    "female_athlete_percentage", "prev_total_medals",
    "prev_medals_per_athlete", "Gold", "Silver",
    "Bronze", "total_medals", "medals_per_athlete"
]

country_year = country_year[final_cols].sort_values(["NOC", "Year"]).reset_index(drop=True)

#save
print(country_year.head(10))
country_year.to_csv("olympics_country_year_features.csv", index=False)
print("\nSaved to olympics_country_year_features.csv")