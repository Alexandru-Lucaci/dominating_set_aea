import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


data = """id,cyan,time
NYP,0,859
HDR,1,462
CJK,0,559
LYN,1,971
ZGQ,0,742
SVX,1,810
JOR,0,675
JFD,1,469
PDH,0,845
YHY,1,450
BQG,0,906
MND,1,573
FIB,0,592
LDS,1,833
RJQ,0,411
DAK,1,846
MGD,1,652
PBF,0,695
WAP,1,291
MQN,0,783
IAD,1,875
AIW,0,700
KDR,1,487
AMO,0,676
SNQ,1,657
TLZ,0,392
ZFP,1,472
YEY,0,556
MLN,1,618
OJN,0,486
YOL,1,779
JNM,0,833
XCE,1,647
NRW,0,844
MTS,1,1005"""

import io
df = pd.read_csv(io.StringIO(data))

df['background'] = df['cyan'].map({0: 'yellow', 1: 'cyan'})

print("First few rows of the dataset:")
print(df.head())

print("\nSummary statistics by background color:")
summary = df.groupby('background')['time'].describe()
print(summary)


yellow_times = df[df['background'] == 'yellow']['time']
cyan_times = df[df['background'] == 'cyan']['time']

yellow_mean = yellow_times.mean()
cyan_mean = cyan_times.mean()
yellow_std = yellow_times.std()
cyan_std = cyan_times.std()

print(f"\nYellow background mean reaction time: {yellow_mean:.2f} ms")
print(f"Cyan background mean reaction time: {cyan_mean:.2f} ms")
print(f"Yellow background standard deviation: {yellow_std:.2f} ms")
print(f"Cyan background standard deviation: {cyan_std:.2f} ms")


t_stat, p_value = stats.ttest_ind(yellow_times, cyan_times, equal_var=False)
print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")


alpha = 0.05
if p_value < alpha:
    print(f"The p-value ({p_value:.4f}) is less than alpha ({alpha}), so we reject the null hypothesis.")
    if yellow_mean < cyan_mean:
        print("Yellow background results in significantly faster reaction times.")
    else:
        print("Cyan background results in significantly faster reaction times.")
else:
    print(f"The p-value ({p_value:.4f}) is greater than alpha ({alpha}), so we fail to reject the null hypothesis.")
    print("There is no significant difference in reaction times between the two background colors.")


n1 = len(yellow_times)
n2 = len(cyan_times)
df_welch = ((yellow_std**2/n1 + cyan_std**2/n2)**2) / ((yellow_std**2/n1)**2/(n1-1) + (cyan_std**2/n2)**2/(n2-1))
t_critical = stats.t.ppf(1 - alpha/2, df_welch)
margin_of_error = t_critical * np.sqrt(yellow_std**2/n1 + cyan_std**2/n2)
ci_lower = (yellow_mean - cyan_mean) - margin_of_error
ci_upper = (yellow_mean - cyan_mean) + margin_of_error

print(f"\n95% Confidence Interval for difference in means (Yellow - Cyan): ({ci_lower:.2f}, {ci_upper:.2f}) ms")



plt.figure(figsize=(10, 6))
means = [yellow_mean, cyan_mean]
errors = [yellow_std / np.sqrt(n1), cyan_std / np.sqrt(n2)]
plt.bar(['Yellow', 'Cyan'], means, yerr=errors, capsize=10, color=['gold', 'cyan'], alpha=0.7)
plt.title('Mean Reaction Times by Background Color with Standard Error')
plt.ylabel('Mean Reaction Time (ms)')
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.show()