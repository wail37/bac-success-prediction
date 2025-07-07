import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# تحديد المسار النسبي للملف
DATA_PATH = "C:/Users/PC/Desktop/bac_success_prediction/data/bac_success_rates_2013_2023.csv"


# قراءة البيانات
df = pd.read_csv(DATA_PATH)

# طباعة نظرة سريعة
print(df.describe())
print(df.head())

# رسم نسبة النجاح
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(x="Year", y="Success_Rate", data=df, marker="o", label="نسبة النجاح")

# خط الاتجاه
z = np.polyfit(df["Year"], df["Success_Rate"], 1)
p = np.poly1d(z)
plt.plot(df["Year"], p(df["Year"]), "--", color="red", label="Trend Line")

plt.title("تطور نسبة النجاح في البكالوريا (2013 - 2023)")
plt.xlabel("السنة")
plt.ylabel("نسبة النجاح (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
