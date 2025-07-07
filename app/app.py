import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# تحميل البيانات
data = pd.read_csv("C:/Users/PC/Desktop/bac_success_prediction/data/bac_success_rates_2013_2023.csv")


# تدريب النموذج
X = data[["Year"]]
y = data["Success_Rate"]
model = LinearRegression()
model.fit(X, y)

# واجهة Streamlit
st.set_page_config(page_title="تحليل نسبة النجاح في البكالوريا", layout="centered")

st.title("📊 تحليل وتوقع نسبة النجاح في البكالوريا الجزائرية")
st.markdown("مشروع باستخدام الذكاء الاصطناعي لتحليل وتوقع نسب النجاح من 2013 إلى 2025")

# رسم المنحنى
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x="Year", y="Success_Rate", data=data, marker="o", ax=ax, label="نسبة النجاح")
z = np.polyfit(data["Year"], data["Success_Rate"], 1)
p = np.poly1d(z)
ax.plot(data["Year"], p(data["Year"]), "--", color="red", label="Trend Line")
ax.set_title("تطور نسبة النجاح في البكالوريا")
ax.set_xlabel("السنة")
ax.set_ylabel("نسبة النجاح (%)")
ax.legend()
st.pyplot(fig)

# عرض التوقعات
future_years = pd.DataFrame({"Year": [2024, 2025]})
preds = model.predict(future_years)
st.subheader("🔮 التوقعات للسنوات القادمة:")
for year, pred in zip(future_years["Year"], preds):
    st.write(f"🟢 {year}: {pred:.2f}%")

# إدخال يدوي
st.subheader("📌 أدخل سنة لتوقع نسبة النجاح:")
input_year = st.number_input("أدخل السنة", min_value=2024, max_value=2030, step=1)
if input_year:
    pred = model.predict([[input_year]])[0]
    st.success(f"🔮 نسبة النجاح المتوقعة في {input_year} هي: {pred:.2f}%")
