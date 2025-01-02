from ydata_profiling import ProfileReport
import pandas as pd
data= pd.read_csv('C:/Users/Taran/Desktop/Student Sleep Pattern/student_sleep_patterns.csv')
print(data.head())
report=ProfileReport(data,title='Sleep Pattern')
report.to_file("Student_Sleep_Pattern.html")