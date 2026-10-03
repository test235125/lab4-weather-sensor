import pandas as pd  # type: ignore
import numpy as np   # type: ignore
from scipy import stats  # type: ignore
data = {
    "Hour": range(1, 16),
    "Temperature": [
        22, 23, np.nan, np.nan, np.nan,
        25, 26, 58, 24, 23,
        22, 21, 23, 24, 25
    ]
}
df = pd.DataFrame(data)
print("Stats BEFORE cleaning:")
print("Mean:", df["Temperature"].mean())
print("Std:", df["Temperature"].std())
print("Skewness:", stats.skew(df["Temperature"].dropna()))
print("Kurtosis:", stats.kurtosis(df["Temperature"].dropna()))
raw_temperature = df["Temperature"].copy()
df["Temperature"] = df["Temperature"].interpolate(method="linear")
print("\nAfter interpolation, missing values:",
      df["Temperature"].isnull().sum())
z_scores = stats.zscore(df["Temperature"])
df["Zscore"] = z_scores
outlier = df["Zscore"].abs() > 2
print("\nZ-score outlier detected at Hour:")
print(df.loc[outlier, "Hour"].tolist())
print("Outlier value:")
print(df.loc[outlier, "Temperature"].tolist())
median_temperature = raw_temperature.median()
df.loc[outlier, "Temperature"] = median_temperature
print("\nReplaced with column median:", median_temperature)
df["Temperature_Zscore"] = stats.zscore(df["Temperature"])
comparison = pd.DataFrame({
    "Hour": df["Hour"],
    "Temp_raw": raw_temperature,
    "Temp_clean": df["Temperature"],
    "Temp_Zscore": df["Temperature_Zscore"]
})
print("\nBefore/After comparison:")
print(comparison)
print("\nStats AFTER cleaning:")
print("Mean:", df["Temperature"].mean())
print("Std:", df["Temperature"].std())