import pandas as pd

# 데이터 로드
file_name = "2019_kbo_for_kaggle_v2.csv"
df = pd.read_csv(file_name)


# 문제 1: 주어진 각 카테고리에서의 탑 10 선수들 from 2015 to 2018
print("\n##########TOP10##########")

# for each year from 2015 to 2018
years = range(2015, 2019)

for year in years:
    print(f"\n <YEAR - {year}>:")
    year_df = df[df['year'] == year]
    # year열에서 값이 year인 행들만 선택

    # Hits
    H_top10 = year_df.nlargest(10, 'H') # 'H'를 기준으로 상위 10개 가져오기
    print(f"\nTop 10 Players in Hits ({year}):")
    print(H_top10[['batter_name', 'H']])

    # Batting Average
    avg_top10 = year_df.nlargest(10, 'avg')
    print(f"\nTop 10 Players in Batting Average ({year}):")
    print(avg_top10[['batter_name', 'avg']])

    # Homerun
    HR_top10 = year_df.nlargest(10, 'HR')
    print(f"\nTop 10 Players in Homerun ({year}):")
    print(HR_top10[['batter_name', 'HR']])

    # On-base Percentage
    OBP_top10 = year_df.nlargest(10, 'OBP')
    print(f"\nTop 10 Players in On-base Percentage ({year}):")
    print(OBP_top10[['batter_name', 'OBP']])


# 문제 2: 승리기여도가 가장 높은 선수 출력 in 2018
print("\n\n##########Highest War##########")
print("\nPlayer with the highest war by position in 2018:")

# 2018년도에 해당하는 행들만 선택 -> 포지션('cp')을 기준으로 그룹화 -> 각 그룹에서 'war'열의 값이 가장 큰 행의 인덱스 찾기
war_max = df[df['year'] == 2018].groupby('cp')['war'].idxmax()

print(df.loc[war_max, ['batter_name', 'cp', 'war']])



# 문제 3: Find the feature with the highest correlation with salary
print("\n\n##########Highest Correlation with Salary##########")

features = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']

# 상관 관계 행렬 계산
corr_mat = df[features].corr()

# 'salary'와의 상관 관계만 선택
corr_with_salary = corr_mat['salary']

# 'salary'를 제외하고 가장 높은 상관 관계를 가진 카테고리를 선택
highest_corr_feat = corr_with_salary.drop('salary').idxmax()

print(f"\nFeature, the highest correlation with salary: {highest_corr_feat}")