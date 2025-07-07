import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# تحميل البيانات
df = pd.read_csv("C:/Users/PC/Desktop/bac_success_prediction/data/bac_success_rates_2013_2023.csv")

# تجهيز البيانات للنموذج
X = df[["Year"]]
y = df["Success_Rate"]

# تدريب النموذج
model = LinearRegression()
model.fit(X, y)

# التوقع للسنوات القادمة
future_years = pd.DataFrame({"Year": [2024, 2025]})
predictions = model.predict(future_years)

# حفظ التوقعات في ملف CSV
output_df = future_years.copy()
output_df["Predicted_Success_Rate"] = predictions

# تحديد المسار الكامل لحفظ النتائج
output_path = "C:/Users/PC/Desktop/bac_success_prediction/outputs/predictions.csv"
output_df.to_csv(output_path, index=False)

# عرض النتائج
print("✅ تم حفظ التوقعات في ملف predictions.csv بنجاح.\n")
print(output_df)

