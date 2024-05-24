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



def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.


df = pd.read_csv('df1.csv')
inv = pd.read_csv('RemainingInventory.csv')
items = pd.read_csv('ItemList.csv')
fcst = pd.read_csv('forecast.csv')
err = pd.read_csv('errors.csv')
df['date'] = df.date.astype('datetime64[ns]')
item = pd.read_csv('item_fc.csv')
item = item.item_name.unique()

st.title("Номин Юнайтед Хайпермаркетын барааны борлуулалт тооцоолох загварын практикал тест")
st.write("2 жилийн (2022-2023 он) хоорондын борлуулалтын үр дүн дээр суурилсан")


item_choices = st.selectbox('Бараа сонгох:',item)
item_info = df[df['item_name']==item_choices][['item_key','item_name','group','category','brand','vendor','base_price']]
item_info = item_info.groupby(['item_key','item_name','group','category','brand','vendor']).mean().reset_index()

df_sum = df.groupby(['item_name']).sum(numeric_only=True).reset_index()
df_avg = df.groupby(['item_name']).mean(numeric_only=True).reset_index()
eda = df[df['item_name'].isin(item.tolist())].set_index('date')[['item_name','item_key','base_price','new_price','amt','qty','on_sale']].copy()
inv = inv.rename(columns={'Iteminfid':'item_key', 'Ognoo':'date'})
inv.date = inv.date.astype('datetime64[ns]')
eda.index = eda.index.astype('datetime64[ns]')
eda = eda.merge(inv, how='left', on=['item_key','date'])
eda = eda.groupby(['date','item_name','base_price']).sum().reset_index().set_index('date')[['item_name','base_price','qty','Uldegdel']]
eda.index = eda.index.astype('datetime64[ns]')
eda = eda.groupby(['date','item_name']).agg({'base_price':'first','qty':'sum','Uldegdel':'sum'}).reset_index()
dr = pd.DataFrame(pd.date_range(start='2022-01-01', end='2023-10-01',freq='D'))
dr.columns = ['date']
df_array = []

for i in item:
    dr = pd.DataFrame(pd.date_range(start='2022-01-01', end='2023-10-01',freq='D'))
    dr.columns = ['date']
    x = dr.merge(eda[eda.item_name == i], how='left',on='date')
    x.item_name = x.item_name.fillna(value=i)
    x.qty = x.qty.fillna(value=0)
    df_array.append(x)

result_df = pd.concat(df_array).reset_index()
eda = result_df.copy()
eda = eda.drop(columns='index')
eda['base_price'] = eda['base_price'].fillna(method='bfill')
eda['Uldegdel'] = eda['Uldegdel'].fillna(method='bfill')
eda['date'] = eda['date'].astype('datetime64[ns]')
eda = eda.set_index('date')
eda['Uldegdel'] = eda['Uldegdel'].abs()
eda['Uldegdel'] = eda['Uldegdel'].apply(lambda x: 1 if x > 0 else 0)
eda_scaled = eda.reset_index()
eda_scaled['date'] = eda_scaled['date'].astype('datetime64[ns]')
eda_scaled = eda_scaled.groupby(['item_name',pd.Grouper(key='date',freq='W')]).agg({'base_price':'median','qty':'sum','Uldegdel':'sum'}).reset_index()
eda = eda.reset_index()
eda['date'] = eda.date.astype('datetime64[ns]')
eda["weekday"] = eda['date'].dt.dayofweek
eda["weekend"] = eda['date'].dt.dayofweek > 4

weekday = eda.groupby(['item_name','weekday']).median().reset_index().set_index('weekday')[['item_name','qty']]


sale = df[['item_name','group','category','brand','vendor']]
df = df.merge(sale.groupby(['item_name','group','category','brand','vendor']).sum().reset_index(), on='item_name', how='left')
avg_salary1 = {
    'date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01', '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01'],
    'national_avg_salary': [1259700, 1259700, 1259700, 1261300, 1261300, 1261300, 1269100, 1269100, 1269100, 1328100, 1328100, 1328100, 1314900, 1314900, 1314900, 1330400, 1330400, 1330400, 1335100, 1335100, 1335100, 1438200, 1438200, 1438200, 1450100, 1450100, 1450100, 1544400, 1544400, 1544400, 1573100, 1573100, 1573100, 1750200, 1750200, 1750200, 1830790, 1830790, 1830790, 1890000, 1890000, 1890000, 2009300, 2009300, 2009300],
    'person_avg_salary': [971000, 971000, 971000, 1018100, 1018100, 1018100, 1077400, 1077400, 1077400, 1038400, 1038400, 1038400, 1056600, 1056600, 1056600, 1092800, 1092800, 1092800, 1145500, 1145500, 1145500, 1200300, 1200300, 1200300, 1197100, 1197100, 1197100, 1325600, 1325600, 1325600, 1361400, 1361400, 1361400, 1469800, 1469800, 1469800, 1516000, 1516000, 1516000, 1601000, 1601000, 1601000, 1662500, 1662500, 1662500]
}

avg_salary = pd.DataFrame(avg_salary1)
cpi_data = {
    'date': ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12', '2024-01'],
    'general_cpi': [0.8, 1.2, 0.6, 2.1, 0.9, 0.9, 1.5, 0.8, 0.3, 0.7, 1, 2.1, 2, 1.3, 0.9, 2.2, 1.3, 1.5, 1, -0.9, -0.2, 1.3, 1.1, 1, 1.2, 1.2, 1, 1.3, 1.3, 1, -0.3, -0.2, -0.2, 0.3, 0.8, 0.4, 0.8],
    'foods_cpi': [2.7, 3.8, 1.5, 4.4, 1.9, 1.2, -0.2, -2.3, -0.8, 2.7, 1.6, 3.1, 3.5, 1.2, 1.3, 3.5, 2.7, 2.5, 1.7, -5.5, -2.2, 2.1, 2, 1.9, 2.2, 3.1, 2.3, 3.3, 3.9, 2.3, -1.5, -3.9, -1.4, -0.2, 0.7, 0.9, 1.8]
}

cpi = pd.DataFrame(cpi_data)
cpi['date'] = pd.to_datetime(cpi['date'] + '-01')
avg_salary['date'] = avg_salary['date'].astype('datetime64[ns]')
env_feats = cpi.merge(avg_salary,on='date', how='left')
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Day_Temperature': ['-16', '-11', '-4', '8', '15', '22', '25', '21', '16', '8', '-11', '-13'],
    'Night_Temperature': ['-26', '-24', '-14', '-4', '4', '6', '12', '10', '4', '-6', '-15', '-22']
}
data_extended = []
for year in range(2021, 2024):
    for month, day_temp, night_temp in zip(data['Month'], data['Day_Temperature'], data['Night_Temperature']):
        data_extended.append({'Date': f'{year}-{month}-01', 'Day_Temperature': day_temp, 'Night_Temperature': night_temp})
df1 = pd.DataFrame(data_extended)
df1['date'] = df1['Date'].astype('datetime64[ns]')
env_feats = env_feats.merge(df1, on='date', how='left')
env_feats = env_feats.drop(columns='Date')
df['date'] = df.date.astype('datetime64[ns]')
df = df.merge(env_feats,on='date',how='left')

arr = []

for i in item:
    temp_df = df[df['item_name']==i]
    temp_df['general_cpi'] = temp_df['general_cpi'].ffill()
    temp_df['foods_cpi'] = temp_df['foods_cpi'].ffill()
    temp_df['national_avg_salary'] = temp_df['national_avg_salary'].ffill()
    temp_df['person_avg_salary'] = temp_df['person_avg_salary'].ffill()
    temp_df['Day_Temperature'] = temp_df['Day_Temperature'].ffill()
    temp_df['Night_Temperature'] = temp_df['Night_Temperature'].ffill()
    temp_df['general_cpi'] = temp_df['general_cpi'].bfill()
    temp_df['foods_cpi'] = temp_df['foods_cpi'].bfill()
    temp_df['national_avg_salary'] = temp_df['national_avg_salary'].bfill()
    temp_df['person_avg_salary'] = temp_df['person_avg_salary'].bfill()
    temp_df['Day_Temperature'] = temp_df['Day_Temperature'].bfill()
    temp_df['Night_Temperature'] = temp_df['Night_Temperature'].bfill()
    arr.append(temp_df)
    
result_df = pd.concat(arr).reset_index()
result_df = result_df.drop(columns='index')
start_date = '2023-07-08'
end_date = '2023-07-15'
naadam = pd.date_range(start=start_date, end=end_date)
additional_start_date = '2023-02-03'
additional_end_date = '2023-02-12'
tsagaan = pd.date_range(start=additional_start_date, end=additional_end_date)
additional_start_date = '2023-12-25'
additional_end_date = '2023-12-31'
newyear = pd.date_range(start=additional_start_date, end=additional_end_date)
additional_start_date = '2023-11-17'
additional_end_date = '2023-11-19'
black = pd.date_range(start=additional_start_date, end=additional_end_date)
def date_features(df):
    dataset = df
    df['year'] = dataset.date.dt.year
    df['month'] = dataset.date.dt.month
    df['day'] = dataset.date.dt.day
    df['season'] = dataset.date.dt.quarter
    df['dayofyear'] = dataset.date.dt.dayofyear
    df['weekofyear'] = dataset.date.dt.isocalendar().week
    df['day^year'] = np.log((np.log(dataset['dayofyear'] + 1)) ** (dataset['year'] - 2000))
    return df
start_date = '2022-07-08'
end_date = '2022-07-15'
naadam1 = pd.date_range(start=start_date, end=end_date)
additional_start_date = '2022-02-03'
additional_end_date = '2022-02-12'
tsagaan1 = pd.date_range(start=additional_start_date, end=additional_end_date)
additional_start_date = '2022-12-25'
additional_end_date = '2022-12-31'
newyear1 = pd.date_range(start=additional_start_date, end=additional_end_date)
additional_start_date = '2022-11-17'
additional_end_date = '2022-11-19'
black1 = pd.date_range(start=additional_start_date, end=additional_end_date)
naadam = naadam.append(naadam1)
tsagaan = naadam.append(tsagaan1)
newyear = naadam.append(newyear1)
black = naadam.append(black1)
result_df['black_friday'] = result_df['date'].isin(black)
result_df['tsagaan'] = result_df['date'].isin(tsagaan)
result_df['naadam'] = result_df['date'].isin(naadam)
result_df['newyear'] = result_df['date'].isin(newyear)

result_df['Day_Temperature'] = result_df['Day_Temperature'].astype(int)
result_df['Night_Temperature'] = result_df['Night_Temperature'].astype(int)

date_features(result_df)


with open('base_rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)
def generate_bar_chart(data, title):
    bar = (
        Bar()
        .add_xaxis(data.index.tolist())
        .add_yaxis("", data.tolist(), itemstyle_opts=opts.ItemStyleOpts(color="red"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"), splitline_opts=opts.SplitLineOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"), splitline_opts=opts.SplitLineOpts(is_show=False)),
            legend_opts=opts.LegendOpts(is_show=False)
        )
    )
    return bar

# Display content based on active tab

bt = st.button('Процесс хийх')
if bt:
    st.header(f"{item_choices}")
    st.divider()
    con = st.container()
    # Assuming your DataFrame is named result_df
    temp_1_1 = result_df[result_df['item_name']==item_choices]
    columns = ['date', 'general_cpi', 'foods_cpi', 'national_avg_salary', 'person_avg_salary', 'Day_Temperature', 'Night_Temperature']
    data = temp_1_1[columns]

    # Convert the 'date' column to datetime if it's not already
    data['date'] = pd.to_datetime(data['date'])

    # Set up Line charts for each pair of variables
    line_general_foods = (
        Line()
        .add_xaxis(data['date'].dt.strftime('%Y-%m-%d').tolist())
        .add_yaxis("Ерөнхий Хэрэглээний", data['general_cpi'].tolist(), linestyle_opts=opts.LineStyleOpts(color="white", width=1, type_="solid"),symbol="none",itemstyle_opts=opts.ItemStyleOpts(color="white"))
        .add_yaxis("Хүнсний", data['foods_cpi'].tolist(), linestyle_opts=opts.LineStyleOpts(color="red", width=2, type_="dotted"),symbol="none",itemstyle_opts=opts.ItemStyleOpts(color="red"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title=""),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),splitline_opts=opts.SplitLineOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),splitline_opts=opts.SplitLineOpts(is_show=False)),
        )
    )

    line_salary = (
        Line()
        .add_xaxis(data['date'].dt.strftime('%Y-%m-%d').tolist())
        .add_yaxis("Улсын дундаж", data['national_avg_salary'].tolist(), linestyle_opts=opts.LineStyleOpts(color="white", width=1, type_="solid"),symbol="none",itemstyle_opts=opts.ItemStyleOpts(color="white"))
        .add_yaxis("Хувь хүний", data['person_avg_salary'].tolist(), linestyle_opts=opts.LineStyleOpts(color="red", width=1, type_="dotted"),symbol="none",itemstyle_opts=opts.ItemStyleOpts(color="red"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title=""),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),splitline_opts=opts.SplitLineOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),splitline_opts=opts.SplitLineOpts(is_show=False)),
        )
    )
#a
    line_temperature = (
        Line()
        .add_xaxis(data['date'].dt.strftime('%Y-%m-%d').tolist())
        .add_yaxis("Өдрийн температур", data['Day_Temperature'].tolist(), linestyle_opts=opts.LineStyleOpts(color="white", width=1, type_="solid"),symbol="none",itemstyle_opts=opts.ItemStyleOpts(color="white"))
        .add_yaxis("Шөнийн температур", data['Night_Temperature'].tolist(), linestyle_opts=opts.LineStyleOpts(color="red", width=1, type_="dotted"),symbol="none",itemstyle_opts=opts.ItemStyleOpts(color="red"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title=""),
            
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),splitline_opts=opts.SplitLineOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),splitline_opts=opts.SplitLineOpts(is_show=False)),
        )
    )



    # Add charts to the container
    with con:
        st.write("### Ерөнхий индикаторууд")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Ерөнхий Хэрэглээний Индекс vs. Хүнсний Индекс")

            st_pyecharts(line_general_foods)
        with col2:
            st.write("Улсын дундаж цалин vs. Хувь хүний дундаж цалин")

            st_pyecharts(line_salary)
        with col3:
            st.write("Өдрийн температур vs. Шөнийн температур")
            st_pyecharts(line_temperature)
    st.divider()
    st.subheader('Барааны мэдээлэл')
    tab1_col1, tab1_col2 = st.columns(2)
    with tab1_col1:  
        st.write(f":red[Груп:] {item_info[item_info['item_name']==item_choices]['group'].values[0]}")
        st.write(f":red[Ангилал:] {item_info[item_info['item_name']==item_choices]['category'].values[0]}")
        st.write(f":red[Бренд:] {item_info[item_info['item_name']==item_choices]['brand'].values[0]}")
        st.write(f":red[Вендор:] {item_info[item_info['item_name']==item_choices]['vendor'].values[0]}")
    with tab1_col2:
        t1c2_total_amt = '₮'+format(df_sum[df_sum['item_name']==item_choices]['amt'].values[0],',.2f')
        st.write(f":red[Нийт борлуулалтын дүн:] {t1c2_total_amt}")
        t1c2_total_sale = format(df_sum[df_sum['item_name']==item_choices]['qty'].values[0],',.2f')
        st.write(f":red[Нийт борлуулсан:] {t1c2_total_sale}")
        t1c2_avg_price = '₮'+format(df_avg[df_avg['item_name']==item_choices]['base_price'].values[0],',.2f')
        st.write(f":red[Дундаж үнэ:] {t1c2_avg_price}")
        t1c2_avg_sale = format(df_avg[df_avg['item_name']==item_choices]['qty'].values[0],',.2f')
        st.write(f":red[Өдрийн дундаж борлуулалтын тоо:] {t1c2_avg_sale}")
    st.divider()
    tab1_col3, tab1_col4 = st.columns(2)
    
    with tab1_col3:
        st.subheader(f":white[Борлуулалтын тоо:]")
        temp_1_1 = result_df[result_df['item_name']==item_choices]
        temp_1_1 = temp_1_1.set_index('date')
        sales_monthly_1 = temp_1_1.resample('M').sum()
        sales_monthly_1.index = sales_monthly_1.index.strftime('%Y-%m')
        line = (
            Line()
            .add_xaxis(sales_monthly_1.index.tolist())  # Convert the index to a list
            .add_yaxis(series_name='Борлуулалтын тоо', y_axis=sales_monthly_1['qty'].tolist(), is_smooth=True,
                       linestyle_opts=opts.LineStyleOpts(color="red"),
                       label_opts=opts.LabelOpts(is_show=False),
                       symbol="none",  # Remove data point symbols
                       itemstyle_opts=opts.ItemStyleOpts(color="red"))
            .set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                                      axisline_opts=opts.AxisLineOpts(is_show=False),
                                                      axistick_opts=opts.AxisTickOpts(is_show=False),
                                                      splitline_opts=opts.SplitLineOpts(is_show=False)),
                             yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                                      axisline_opts=opts.AxisLineOpts(is_show=False),
                                                      axistick_opts=opts.AxisTickOpts(is_show=False),
                                                      splitline_opts=opts.SplitLineOpts(is_show=False)),
                             tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                                                           formatter=opts.TooltipOpts(formatter='{b}: {c}')))
        )
        st_pyecharts(line)
        
        
        #fig = px.line(sales_monthly_1['qty'],line_shape="spline")
        #fig.update_traces(line_color='#FF0000',line_width=2.4,showlegend=False)
        #fig.update_layout(xaxis_title='Огноо', yaxis_title='Тоо')
        #st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f":white[7 хоногийн өдөр тус бүрийн дундаж борлуулалт:]")
        temp_1_2 = weekday[weekday['item_name']==item_choices].copy()
        
        bar = (
        Bar()
        .add_xaxis(temp_1_2.index.tolist())  # Convert the index to a list
        .add_yaxis(series_name='Борлуулалтын тоо', y_axis=temp_1_2['qty'].tolist(),
               label_opts=opts.LabelOpts(is_show=False),  # Remove data point symbols
               itemstyle_opts=opts.ItemStyleOpts(color="red"))
        .set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                              axisline_opts=opts.AxisLineOpts(is_show=False),
                                              axistick_opts=opts.AxisTickOpts(is_show=False),
                                              splitline_opts=opts.SplitLineOpts(is_show=False)),
                     yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                              axisline_opts=opts.AxisLineOpts(is_show=False),
                                              axistick_opts=opts.AxisTickOpts(is_show=False),
                                              splitline_opts=opts.SplitLineOpts(is_show=False)),
                     tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                                                   formatter=opts.TooltipOpts(formatter='{b}: {c}')),
                        legend_opts=opts.LegendOpts(is_show=False))
        )
        st_pyecharts(bar)
        
        #fig = px.bar(temp_1_2['qty'],text = temp_1_2["qty"])
        #fig.update_traces(marker_color='#FF0000',showlegend=False)
        #fig.update_layout(xaxis_title='Гараг', yaxis_title='Тоо')
        #st.plotly_chart(fig,use_container_width=True)
        
    with tab1_col4:
        st.subheader(f":white[Борлуулалтын тренд:]")
        # Assuming eda is your DataFrame and item_choices is a list of item names
        k_1 = eda[eda['item_name'] == item_choices]
        k_1 = k_1.set_index('date')
        k_1 = k_1.resample('M').mean(numeric_only=True)
        numeric_dates_1 = (k_1.index - k_1.index[0]).days
        coefficients = np.polyfit(numeric_dates_1, k_1['qty'], 1)
        trendline_x = numeric_dates_1  # Use original x-axis data for the trendline
        trendline_y = np.polyval(coefficients, numeric_dates_1)  # Calculate predicted y-values for the trendline

        # Convert datetime index to strings with format 'YYYY-MM'
        x_labels = k_1.index.strftime('%Y-%m').tolist()

        # Create the Scatter chart
        scatter = (
            Scatter()
            .add_xaxis(x_labels)  # Set x-axis data with formatted dates
            .add_yaxis("Борлуулалтын тоо", k_1['qty'].tolist(), label_opts=opts.LabelOpts(is_show=False), symbol="circle", 
                       symbol_size=8, itemstyle_opts=opts.ItemStyleOpts(color="red"))  # Set y-axis data with a series name and red color
        )

        # Create the Line chart for the trendline
        line = (
            Line()
            .add_xaxis(x_labels)  # Set x-axis data for the trendline with formatted dates
            .add_yaxis("Хандлагын шугам", trendline_y.tolist(), 
                       linestyle_opts=opts.LineStyleOpts(color="white", type_='dotted',width=0.5),  # Turn the line red
                       symbol="none",itemstyle_opts=opts.ItemStyleOpts(color="white"))  # Remove markers
        )

        # Combine the Scatter and Line charts
        scatter.overlap(line)
        #a
        # Set global options for the chart
        scatter.set_global_opts(
                                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                                         axisline_opts=opts.AxisLineOpts(is_show=False),
                                                         axistick_opts=opts.AxisTickOpts(is_show=False),
                                                         splitline_opts=opts.SplitLineOpts(is_show=False)),
                                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                                         axisline_opts=opts.AxisLineOpts(is_show=False),
                                                         axistick_opts=opts.AxisTickOpts(is_show=False),
                                                         splitline_opts=opts.SplitLineOpts(is_show=False)),
                                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                                                              formatter=opts.TooltipOpts(formatter='{b}: {c}'))
                               )

        st_pyecharts(scatter)
        
        
        #fig = px.scatter(k_1['qty'])
        #fig.add_scatter(x=k_1.index, y=trendline_1, mode='lines', showlegend=False)
        #fig.update_traces(marker_color='#FF0000',showlegend=False)
        #fig.update_layout(xaxis_title='Гараг', yaxis_title='Тоо')
        #st.plotly_chart(fig, use_container_width=True)
        
        
        
        st.subheader(f":white[Борлуулагдах тоо хэмжээний дистрибюшн:]")
        f_1 = eda[eda['item_name']==item_choices]
        f_1 = f_1.groupby(['date','item_name']).sum('qty').reset_index()
        fig = px.violin(f_1['qty'])
        fig.update_traces(marker_color='#FF0000',showlegend=False)
        fig.update_layout(xaxis_title='',yaxis_title='',height=350)
        st.plotly_chart(fig, use_container_width=True)
        
    st.divider()
    st.subheader(f"Корреляци")
    exp_1 = result_df[result_df.item_name==item_choices]
    fig = px.imshow(exp_1.drop(columns=exp_1.columns[0]).corr(numeric_only=True),color_continuous_scale=['white', 'blue'])
    st.plotly_chart(fig,use_container_width=True)
    st.divider()
    st.write('Прогноз')
    fcst_1 = fcst[fcst.item_name==item_choices].set_index('date')[['predicted_quantity','qty','preds']]
    fcst_1 = fcst_1.rename(columns={'predicted_quantity':'AUTO',
                    'qty':'REAL',
                    'preds':'ML'})
    forecast1 = (
    Line()
    .add_xaxis(fcst_1.index.tolist())
    .add_yaxis("Автомат захиалга", fcst_1['AUTO'].tolist(), linestyle_opts=opts.LineStyleOpts(color="orange", type_="dashed"),symbol="none", label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="orange"))
    .add_yaxis("Бодит", fcst_1['REAL'].tolist(), linestyle_opts=opts.LineStyleOpts(color="red", type_="solid", width=2),symbol="none", label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="red"))
    .add_yaxis("Загвар", fcst_1['ML'].tolist(), linestyle_opts=opts.LineStyleOpts(color="white", type_="dashed"),symbol="none", label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="white"))
    .set_global_opts(
        title_opts=opts.TitleOpts(title=""),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"), splitline_opts=opts.SplitLineOpts(is_show=False)),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"), splitline_opts=opts.SplitLineOpts(is_show=False)),
        legend_opts=opts.LegendOpts(is_show=True),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                                                           formatter=opts.TooltipOpts(formatter='{b}: {c}'))))

    
    st_pyecharts(forecast1)
    lossmetrics = err[err.item_name==item_choices].rename(columns={'auto_loss_amt':'Автомат захиалгын алдагдлын дүн',
                                                                  'ml_loss_amt':'Загварын алдагдлын дүн',
                                                                  'auto_CFE':'Автомат захиалгын нийт алдагдлын тоо',
                                                                  'ml_CFE':'Загварын нийт алдагдлын тоо'})
    lossamt = lossmetrics[['Автомат захиалгын алдагдлын дүн', 'Загварын алдагдлын дүн']]
    lossqty = lossmetrics[['Автомат захиалгын нийт алдагдлын тоо', 'Загварын нийт алдагдлын тоо']]
    loss_err = lossmetrics[['auto_RMSE','ml_RMSE','auto_MAE','ml_MAE']]

                # Create bar charts for each variable

    bar_lossamt = generate_bar_chart(lossamt.squeeze(), "")
    st.write("Алдагдлын тоо")
    bar_lossqty = generate_bar_chart(lossqty.squeeze(), "")
    st_pyecharts(bar_lossqty)
    st.write("Зөрүү")
    bar_loss_err = generate_bar_chart(loss_err.squeeze(), "")
    st_pyecharts(bar_loss_err)

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
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(sales_non["ml_qty"], sales_non["qty"])
    mse = mean_squared_error(sales_non["ml_qty"], sales_non["qty"], squared=True)
    st.write(f'Загварын абсолют зөрүү:{mae}')
    st.write(f'Загварын квадрат зөрүү:{mse}')
    mae1 = mean_absolute_error(sales_non["S0"], sales_non["qty"])
    mse1 = mean_squared_error(sales_non["S0"], sales_non["qty"], squared=True)
    st.write(f'Автомат захиалгын абсолют зөрүү:{mae1}')
    st.write(f'Автомат захиалгын квадрат зөрүү:{mse1}')
