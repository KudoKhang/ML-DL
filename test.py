import pandas as pd
import numpy as np

x = np.linspace(30,100,30).reshape(-1,1)
noise = np.random.normal(0,1,30).reshape(-1,1)
y = 15*x + 8 + 20*noise

df = pd.DataFrame({'dien tich': np.linspace(30,100,30),
                   'gia nha': np.linspace(30,100,30)})
df.to_csv('./sources/test.csv', index=False)