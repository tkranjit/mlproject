import pandas as pd
import os
current_path = os.getcwd()
print(current_path)
df=pd.read_csv("../../notebook\data\stud.csv")
print(df.head())