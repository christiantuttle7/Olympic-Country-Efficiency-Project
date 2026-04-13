import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("athlete_events.csv").copy()

# Keep Summer Olympics only
df = df[df["Season"] == "Summer"].copy()

# =========================
# 2. BASIC CLEANING
# =========================
# Medal flags
df["Gold"] = (df["Medal"] == "Gold").astype(int)
df["Silver"] = (df["Medal"] == "Silver").astype(int)
df["Bronze"] = (df["Medal"] == "Bronze").astype(int)
df["AnyMedal"] = df["Medal"].notna().astype(int)

# Female flag
df["Female"] = (df["Sex"] == "F").astype(int)

# Use athlete ID so repeated appearances in multiple events do not inflate athlete counts
# Each row = athlete-event, so aggregation must be careful

# =========================
# 3. HOST COUNTRY MAPPING
# =========================
# You can expand this if your dataset includes more years/cities.
# This maps Summer Olympic YEAR to host NOC.
host_map = {
    1896: "GRE",
    1900: "FRA",
    1904: "USA",
    1906: "GRE",   # Intercalated Games if present
    1908: "GBR",
    1912: "SWE",
    1920: "BEL",
    1924: "FRA",
    1928: "NED",
    1932: "USA",
    1936: "GER",
    1948: "GBR",
    1952: "FIN",
    1956: "AUS",
    1960: "ITA",
    1964: "JPN",
    1968: "MEX",
    1972: "GER",   # West Germany in historical reality; athlete_events uses NOC values
    1976: "CAN",
    1980: "URS",
    1984: "USA",
    1988: "KOR",
    1992: "ESP",
    1996: "USA",
    2000: "AUS",
    2004: "GRE",
    2008: "CHN",
    2012: "GBR",
    2016: "BRA"
}

df["HostNOC"] = df["Year"].map(host_map)
df["host_country"] = (df["NOC"] == df["HostNOC"]).astype(int)

# =========================
# 4. AGGREGATE TO COUNTRY-YEAR
# =========================

# Athletes sent: unique athletes from that NOC in that year
athletes_sent = (
    df.groupby(["NOC", "Year"])["ID"]
    .nunique()
    .reset_index(name="athletes_sent")
)

# Sports participated: unique sports for that NOC in that year
sports_participated = (
    df.groupby(["NOC", "Year"])["Sport"]
    .nunique()
    .reset_index(name="sports_participated")
)

# Events participated: unique events for that NOC in that year
events_participated = (
    df.groupby(["NOC", "Year"])["Event"]
    .nunique()
    .reset_index(name="events_participated")
)

# Female athlete percentage:
# unique athletes by sex within NOC-year, then percentage female
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

# Medal counts by country-year
# Important: team events can create duplicate medals because many athletes on one team get medals.
# This code counts medal-winning ATHLETE entries, not unique team medals.
# If your professor wants official medal table totals, that needs extra deduping logic.
medal_counts = (
    df.groupby(["NOC", "Year"])[["Gold", "Silver", "Bronze"]]
    .sum()
    .reset_index()
)

medal_counts["total_medals"] = (
    medal_counts["Gold"] + medal_counts["Silver"] + medal_counts["Bronze"]
)

# Host flag
host_flag = (
    df.groupby(["NOC", "Year"])["host_country"]
    .max()
    .reset_index()
)

# Merge all olympics-derived features
country_year = athletes_sent.merge(sports_participated, on=["NOC", "Year"], how="outer")
country_year = country_year.merge(events_participated, on=["NOC", "Year"], how="outer")
country_year = country_year.merge(female_pct, on=["NOC", "Year"], how="outer")
country_year = country_year.merge(medal_counts, on=["NOC", "Year"], how="outer")
country_year = country_year.merge(host_flag, on=["NOC", "Year"], how="outer")

# Fill missing medal counts with 0
for col in ["Gold", "Silver", "Bronze", "total_medals", "host_country"]:
    country_year[col] = country_year[col].fillna(0)

# medals_per_athlete
country_year["medals_per_athlete"] = (
    country_year["total_medals"] / country_year["athletes_sent"]
)

# =========================
# 5. PREVIOUS OLYMPICS FEATURES
# =========================
country_year = country_year.sort_values(["NOC", "Year"]).reset_index(drop=True)

country_year["prev_total_medals"] = (
    country_year.groupby("NOC")["total_medals"].shift(1).fillna(0)
)

country_year["prev_medals_per_athlete"] = (
    country_year.groupby("NOC")["medals_per_athlete"].shift(1).fillna(0)
)

# =========================
# 6. MERGE EXTERNAL COLUMNS
# =========================
# Map NOC to ISO3. 
# Note: For strict accuracy, you might want to join your `noc_regions.csv` 
# here if NOC codes differ slightly from standard World Bank ISO3 codes.
country_year["ISO3"] = country_year["NOC"]

# Load Population data
pop_df = pd.read_csv("population.csv")
pop_df = pop_df[["Country Code", "Year", "Value"]].rename(
    columns={"Country Code": "ISO3", "Value": "population"}
)

# Load GDP data
gdp_df = pd.read_csv("gdp.csv")
gdp_df = gdp_df[["Country Code", "Year", "Value"]].rename(
    columns={"Country Code": "ISO3", "Value": "gdp_total"}
)

# Merge Population and GDP into the main country_year dataframe
country_year = country_year.merge(pop_df, on=["ISO3", "Year"], how="left")
country_year = country_year.merge(gdp_df, on=["ISO3", "Year"], how="left")

# Calculate GDP per Capita
country_year["gdp_per_capita"] = country_year["gdp_total"] / country_year["population"]

# Drop gdp_total as we only need gdp_per_capita
country_year = country_year.drop(columns=["gdp_total"])

# Add income_group placeholder (if you want to bring this in later)
country_year["income_group"] = np.nan


# =========================
# 7. REORDER COLUMNS
# =========================
final_cols = [
    "NOC",
    "ISO3",
    "Year",
    "population",
    "gdp_per_capita",
    "income_group",
    "host_country",
    "athletes_sent",
    "sports_participated",
    "events_participated",
    "female_athlete_percentage",
    "prev_total_medals",
    "prev_medals_per_athlete",
    "Gold",
    "Silver",
    "Bronze",
    "total_medals",
    "medals_per_athlete"
]

country_year = country_year[final_cols].sort_values(["NOC", "Year"]).reset_index(drop=True)

# =========================
# 8. SAVE + PREVIEW
# =========================
print(country_year.head(10))
country_year.to_csv("olympics_country_year_features.csv", index=False)
print("\nSaved to olympics_country_year_features.csv")