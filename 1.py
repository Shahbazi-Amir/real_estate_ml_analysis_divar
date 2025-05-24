from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import pandas as pd

options = Options()
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
options.add_argument(f"user-agent={user_agent}")
options.add_argument("--headless")  # بدون باز شدن پنجره مرورگر

driver = webdriver.Chrome(options=options)

base_url = "https://divar.ir/s/tehran/rent-villa"
all_listings = []

for page in range(1, 10):  # گرفتن ۹ صفحه
    print(f"Fetching page {page}...")
    url = f"{base_url}?page={page}"
    driver.get(url)
    time.sleep(5)  # صبر برای لود کامل صفحه

    ads = driver.find_elements(By.CSS_SELECTOR, "article.unsafe-kt-post-card")
    if not ads:
        print("دیگه آگهی بیشتری وجود نداره")
        break

    for ad in ads:
        try:
            title = ad.find_element(By.CSS_SELECTOR, "h2.unsafe-kt-post-card__title").text
            price_elems = ad.find_elements(By.CSS_SELECTOR, "div.unsafe-kt-post-card__description")
            deposit = price_elems[0].text if len(price_elems) > 0 else ""
            rent = price_elems[1].text if len(price_elems) > 1 else ""
            location = ad.find_element(By.CSS_SELECTOR, "span.unsafe-kt-post-card__bottom-description").get_attribute("title")
            link = ad.find_element(By.CSS_SELECTOR, "a.unsafe-kt-post-card__action").get_attribute("href")
            if not link.startswith("http"):
                link = "https://divar.ir" + link

            listing = {
                "title": title,
                "location": location,
                "deposit": deposit,
                "rent": rent,
                "link": link
            }
            all_listings.append(listing)
        except Exception as e:
            print("خطا در خواندن یک آگهی:", e)

driver.quit()

# ذخیره در CSV
df = pd.DataFrame(all_listings)
df.to_csv("divar_listings.csv", index=False)
print(f"\nتعداد کل آگهی‌ها جمع‌آوری شده: {len(df)}")


import pandas as pd
import numpy as np

# خواندن داده از فایل csv
df = pd.read_csv("divar_listings.csv")

# نمایش اولین چند ردیف
df.head()

import re
def persian_to_english_number(text):
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    english_digits = "0123456789"
    translation_table = str.maketrans("".join(persian_digits), "".join(english_digits))
    return text.translate(translation_table)

def clean_price_text(text):
    if pd.isna(text):
        return None
    text = text.replace("٬", "").replace(",", "").replace("تومان", "").replace(" ", "").strip()
    text = persian_to_english_number(text)
    if "رایگان" in text or "توافقی" in text:
        return 0
    digits = re.findall(r"\d+", text)
    if digits:
        return int("".join(digits))
    return None


import pandas as pd
import re

# بارگذاری دوباره داده
df = pd.read_csv("divar_listings.csv")

# جدا کردن اعداد از متون فارسی در ستون deposit و rent
def extract_number(text):
    # پیدا کردن تمام اعداد در متن (با regex)
    numbers = re.findall(r'\d[\d,.]*', text.replace('٬', ','))
    if numbers:
        return float(numbers[0].replace(',', ''))
    return 0  # اگر عددی پیدا نشد، صفر برگردانده شود

# اعمال تابع به ستون‌ها
df['deposit_num'] = df['deposit'].apply(extract_number)
df['rent_num'] = df['rent'].apply(extract_number)

# تبدیل به میلیون تومان
df['deposit_million'] = df['deposit_num'] / 1e6
df['rent_million'] = df['rent_num'] / 1e6

# نمایش داده‌های جدید
df[['deposit', 'deposit_num', 'deposit_million', 'rent', 'rent_num', 'rent_million']].head()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['deposit_million'], df['rent_million'], alpha=0.6)
plt.title("Scatter Plot of Deposit vs Rent (in Millions T)")
plt.xlabel("Deposit (Million T)")
plt.ylabel("Rent (Million T)")
plt.grid(True)
plt.show()



import seaborn as sns

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='deposit_million', y='rent_million', fill=True, cmap='OrRd')
plt.title("Kernel Density Estimate (KDE) Plot")
plt.xlabel("Deposit (Million T)")
plt.ylabel("Rent (Million T)")
plt.grid(True)
plt.show()

# فیلتر کردن بر اساس ودیعه و اجاره
filtered_df = df[(df['deposit_million'] < 2000) & (df['rent_million'] < 30)]

# نمایش آماری و چند نمونه
print(f"تعداد آگهی‌های باقی‌مانده: {len(filtered_df)}")
filtered_df[['title', 'location', 'deposit_million', 'rent_million']].head()



