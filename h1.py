import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import hmac
import plotly.express as px
import seaborn as sns
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pyecharts.options as opts
from scipy.stats import gaussian_kde
from streamlit_echarts import st_pyecharts
from pyecharts.charts import Line, Bar, Scatter, Boxplot

sales_non = pd.read_csv('ordering.csv')
for i in sales_non.name.unique():
    x = sales_non[sales_non.name==i]
    x = x.set_index('date')[['ML_auto','Q_auto']]

    fig = px.scatter(x[['ML_auto','Q_auto']])
    fig.update_layout(xaxis_title=i, yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    #fig.add_scatter(x=k_1.index, y=trendline_1, mode='lines', showlegend=False)
    
    #fig.update_layout(xaxis_title='Гараг', yaxis_title='Тоо')
    #st.plotly_chart(fig, use_container_width=True)

mae = mean_absolute_error(sales_non["ml_qty"], sales_non["qty"])
mse = mean_squared_error(sales_non["ml_qty"], sales_non["qty"], squared=True)
st.write(f'Загварын абсолют зөрүү:{mae}')
st.write(f'Загварын квадрат зөрүү:{mse}')
mae1 = mean_absolute_error(sales_non["S0"], sales_non["qty"])
mse1 = mean_squared_error(sales_non["S0"], sales_non["qty"], squared=True)
st.write(f'Автомат захиалгын абсолют зөрүү:{mae1}')
st.write(f'Автомат захиалгын квадрат зөрүү:{mse1}')
