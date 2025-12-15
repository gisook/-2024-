import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# 1. 데이터 불러오기
# =========================
df = pd.read_csv(
    r'C:\Users\USER\Desktop\오픈소스 과제\archive\product_sales_dataset_final.csv'
)

# 컬럼명 정리
df.columns = df.columns.str.strip().str.replace(' ', '_')

# 날짜형 변환
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%m-%d-%y')

# 파생 변수
df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['Profit_Margin'] = df['Profit'] / df['Revenue']


# =========================
# 2. 계절 컬럼 생성
# =========================
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

df['Season'] = df['Month'].apply(get_season)


# =========================
# 3. groupby 통계 분석
# =========================
category_stats = df.groupby('Category').agg(
    Total_Revenue=('Revenue', 'sum'),
    Avg_Revenue=('Revenue', 'mean'),
    Total_Profit=('Profit', 'sum'),
    Avg_Profit=('Profit', 'mean'),
    Order_Count=('Order_ID', 'count')
)

print("카테고리별 통계")
print(category_stats)


# =========================
# 4. 시각화
# =========================

# (1) 카테고리별 총 매출
category_stats['Total_Revenue'].plot(kind='bar')
plt.title('Total Revenue by Category')
plt.ylabel('Revenue')
plt.xlabel('Category')
plt.tight_layout()
plt.show()

# (2) 월별 매출 추이
monthly_sales = df.groupby('Month')['Revenue'].sum()
monthly_sales.plot()
plt.title('Monthly Revenue Trend')
plt.ylabel('Revenue')
plt.xlabel('Month')
plt.tight_layout()
plt.show()


# =========================
# 5. 머신러닝 (매출 예측)
# =========================
X = df[['Quantity', 'Unit_Price', 'Category', 'Region']]
y = df['Revenue']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['Category', 'Region'])
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n모델 성능")
print("MSE :", mse)
print("RMSE:", rmse)
print("R^2 :", r2)


# =========================
# 6. 지역 + 계절 기준 카테고리 판매 분석
# =========================
season_region_category = (
    df.groupby(['Region', 'Season', 'Category'])
      .agg(
          Total_Quantity=('Quantity', 'sum'),
          Total_Revenue=('Revenue', 'sum')
      )
      .reset_index()
)

# =========================
# 7. 입력 기반 조회 함수
# =========================
def get_category_sales(region, season):
    result = season_region_category[
        (season_region_category['Region'] == region) &
        (season_region_category['Season'] == season)
    ].sort_values(by='Total_Quantity', ascending=False)
    
    return result


# =========================
# 8. 사용 예시 + 시각화
# =========================
region_input = 'West'
season_input = 'Summer'

result = get_category_sales(region_input, season_input)

print(f"\n[{region_input} 지역 - {season_input} 계절] 카테고리별 판매 현황")
print(result)

# 시각화
if not result.empty:
    result.plot(
        x='Category',
        y='Total_Quantity',
        kind='bar',
        legend=False
    )
    plt.title(f'Category Sales ({region_input} - {season_input})')
    plt.ylabel('Total Quantity Sold')
    plt.xlabel('Category')
    plt.tight_layout()
    plt.show()
else:
    print("해당 조건의 데이터가 없습니다.")
