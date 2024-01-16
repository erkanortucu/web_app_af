import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime as dt
import openpyxl
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


import scipy.special
import scipy.misc
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from scipy.special import logsumexp
from sklearn.preprocessing import MinMaxScaler

from lifetimes.plotting import plot_period_transactions
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from itertools import combinations
pd.options.mode.chained_assignment = None

import matplotlib.colors as mcolors
from operator import attrgetter


# streamlit run .\af_analysis\af_data.py


st.set_page_config(layout="wide")


# Functions for each of the pages

def home(fl):
    if fl:
        st.header('Begin exploring the data using the menu on the left')
    else:
        st.header('To begin please upload a file')

    def af_price_data_prep(dataframe):
        dataframe["customer_class"] = dataframe["customer_id"].str[0]

        def grab_col_names(dataframe, cat_th=10, car_th=20):
            """
            Returns the names of categorical, numeric and categorical but cardinal variables in the data set.

            Parameters
            ----------
            dataframe: dataframe
                variable names are the dataframe to be retrieved.
            cat_th: int, float
                class threshold for numeric but categorical variables
            car_th: int, float
                class threshold for categorical but cardinal variables

            Returns
            -------
            cat_cols: list
                Categorical variables list
            num_cols: list
                Numeric variable list
            cat_but_car: list
                List of cardinal variables with categorical view

            Notes
            ------
            cat_cols + num_cols + cat_but_car = total number of variables
            num_but_cat is inside cat_cols.

            """
            # cat_cols, cat_but_car
            cat_cols = [col for col in dataframe.columns if
                        str(dataframe[col].dtypes) in ["category", "object", "bool"]]

            num_but_cat = [col for col in dataframe.columns if
                           dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

            cat_but_car = [col for col in dataframe.columns if
                           dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

            cat_cols = cat_cols + num_but_cat
            cat_cols = [col for col in cat_cols if col not in cat_but_car]

            num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
            num_cols = [col for col in num_cols if col not in cat_cols]

            print(f"Observations: {dataframe.shape[0]}")
            print(f"Variables: {dataframe.shape[1]}")
            print(f'cat_cols: {len(cat_cols)}')
            print(f'num_cols: {len(num_cols)}')
            print(f'cat_but_car: {len(cat_but_car)}')
            print(f'num_but_cat: {len(num_but_cat)}')

            return cat_cols, num_cols, cat_but_car, num_but_cat

        cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)

        cat_cols = cat_cols + cat_but_car

        df = dataframe

        print(df.isnull().sum())
        print(df.info())
        print("################################")
        print(df.head())

        return df

    df = af_price_data_prep(data)
    df2 = df.copy()


    st.write(df.sort_values("registration_date", ignore_index=True))

    st.write("")

    shape1, shape2, shape3 = st.columns(3)

    with shape1:
        st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Total Rows    :  {:,}</h1>".format(
                df.shape[0]), unsafe_allow_html=True)

    with shape2:
        st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Total Columns     :  {:,}</h1>".format(
                df.shape[1]), unsafe_allow_html=True)

    dtypes_df = pd.DataFrame(df.dtypes)
    dtypes_df = dtypes_df.reset_index()
    dtypes_df.columns = ["Variable", "Type"]

    st.markdown(
        "<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Data Types </h1>", unsafe_allow_html=True)

    st.dataframe(dtypes_df)


def data_summary():

    st.header('Reporting & Analysis')


    def af_price_data_prep(dataframe):
        dataframe["customer_class"] = dataframe["customer_id"].str[0]

        def grab_col_names(dataframe, cat_th=10, car_th=20):
            """
            Returns the names of categorical, numeric and categorical but cardinal variables in the data set.

            Parameters
            ----------
            dataframe: dataframe
                variable names are the dataframe to be retrieved.
            cat_th: int, float
                class threshold for numeric but categorical variables
            car_th: int, float
                class threshold for categorical but cardinal variables

            Returns
            -------
            cat_cols: list
                Categorical variables list
            num_cols: list
                Numeric variable list
            cat_but_car: list
                List of cardinal variables with categorical view

            Notes
            ------
            cat_cols + num_cols + cat_but_car = total number of variables
            num_but_cat is inside cat_cols.

            """
            # cat_cols, cat_but_car
            cat_cols = [col for col in dataframe.columns if
                        str(dataframe[col].dtypes) in ["category", "object", "bool"]]

            num_but_cat = [col for col in dataframe.columns if
                           dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

            cat_but_car = [col for col in dataframe.columns if
                           dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

            cat_cols = cat_cols + num_but_cat
            cat_cols = [col for col in cat_cols if col not in cat_but_car]

            num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
            num_cols = [col for col in num_cols if col not in cat_cols]

            print(f"Observations: {dataframe.shape[0]}")
            print(f"Variables: {dataframe.shape[1]}")
            print(f'cat_cols: {len(cat_cols)}')
            print(f'num_cols: {len(num_cols)}')
            print(f'cat_but_car: {len(cat_but_car)}')
            print(f'num_but_cat: {len(num_but_cat)}')

            return cat_cols, num_cols, cat_but_car, num_but_cat

        cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)

        cat_cols = cat_cols + cat_but_car

        df = dataframe

        print(df.isnull().sum())
        print(df.info())
        print("################################")
        print(df.head())

        return df

    df = af_price_data_prep(data)
    df2=df.copy()
    df3 = df.copy()



    kcol1, kcol2 = st.columns((2))
    startDate = pd.to_datetime(df["registration_close_date"]).min()
    endDate = pd.to_datetime(df["registration_close_date"]).max()

    with kcol1:
        date1 = pd.to_datetime(st.date_input(" Start Date  ", startDate))

    with kcol2:
        date2 = pd.to_datetime(st.date_input(" End Date  ", endDate))

    df = df.loc[(df["registration_close_date"] >= date1) & (df["registration_close_date"] <= date2)].copy()

    df2 = df2.loc[(df2["registration_close_date"] >= date1) & (df2["registration_close_date"] <= date2)].copy()
    df3 = df3.loc[(df3["registration_date"] >= date1) & (df3["registration_date"] <= date2)].copy()



    st.write("")

    st.write("")

    st.markdown(
        "<div style='text-align: center;'><h1 style='font-size: 28px;color: navy;'>-------------    Equipment Incoming to The Technical Service    -------------</h1></div>",
        unsafe_allow_html=True)

    st.write("")

    st.markdown("<h1 style='text-align: left; color: black; font-size: 20px; font-weight: bold;'>Number of Equipment Incoming     : {}</h1>".format(
            df3["service_request_number"].count()), unsafe_allow_html=True)

    incclas1, incclas2 = st.columns((2))

    with incclas1:
        st.write("<h1 style='font-size: 20px;color: black;'>Distribution of Incoming Equipment</h1>", unsafe_allow_html=True)

        st.write(df3["equipment_type"].value_counts())

    with incclas2:
        st.write("<h1 style='font-size: 20px;color: black;'>Distribution of Incoming Equipment Class </h1>", unsafe_allow_html=True)

        # Count the occurrences of each equipment class
        equipment_class_counts = df3['equipment_class'].value_counts()
        # Create a new figure and axis
        fig, ax = plt.subplots()
        # Create a bar chart
        ax.bar(equipment_class_counts.index, equipment_class_counts)
        # Set font properties for title
        hfont = {'fontname': 'serif', 'weight': 'bold'}
        #ax.set_title('Equipment Class', size=9, **hfont)
        ax.set_xlabel('Equipment Classes', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_xticklabels(equipment_class_counts.index,rotation=90, fontsize=8)

        # Add percentages on top of the bars
        for i, count in enumerate(equipment_class_counts):
            ax.text(i, count + 0.1, f'{count / len(df) * 100:.1f}%', ha='center', fontsize=7, **hfont)

        # Display the plot in Streamlit
        st.pyplot(fig)

    st.write("")

    customer_class_counts = df3["customer_class"].value_counts()

    # Fontu ayarla
    plt.rcParams['font.family'] = 'serif'

    # customer_class_counts'i kullanarak pasta grafiği oluştur
    fig, ax = plt.subplots(figsize=(8, 4))  # Figür boyutunu belirle
    ax.pie(customer_class_counts, labels=customer_class_counts.index, autopct='%1.1f%%')
    hfont = {'fontname': 'serif', 'weight': 'bold'}
    ax.set_title('Customer Class Incoming', size=9, **hfont)

    # Grafiği sol tarafta göster
    col1, col2, col3, col4 = st.columns(4)  # Ekranı ikiye bölecek

    with col1:
        st.pyplot(fig)
    with col2:
        st.write("")
        st.write("")



        st.write("")

        st.markdown("<h1 style='font-size: 15px;color: black;'> O : Private Companies </h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 15px;color: black;'> K : Public Body </h1>", unsafe_allow_html=True)







    st.write("  ")



    st.markdown("<h1 style='font-size: 20px;color: black;'>Incoming Customer List </h1>", unsafe_allow_html=True)

    kayit_cols = ["registration_date", "customer_name", "customer_id", "equipment_type", "serial_number", "kw", "fan_code",
                  "service_request_number"]
    st.write(df3[kayit_cols].sort_values("registration_date"), ignore_index=True)


    st.write(" ")

    st.markdown("<div style='text-align: center;'><h1 style='font-size: 28px;color: navy;'>-------------    Shipped Equipments    -------------</h1></div>",
        unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: left; color: black; font-size: 20px; font-weight: bold;'> Number of Shipped Equipments     : {}</h1>".format(
            df2["service_request_number"].count()), unsafe_allow_html=True)
    ship1, ship2 = st.columns((2))
    with ship1 :
        st.write("<h1 style='font-size: 20px;color: black;'> Distribution of Shipped Equipment</h1>", unsafe_allow_html=True)
        st.write(df2["equipment_type"].value_counts())

    with ship2:

        st.write("<h1 style='font-size: 20px;color: black;'>Distribution of Shipped Equipment Class </h1>",
                 unsafe_allow_html=True)

        # Count the occurrences of each equipment class
        equipment_class_counts_ship = df2['equipment_class'].value_counts()
        # Create a new figure and axis
        fig, ax = plt.subplots()
        # Create a bar chart
        ax.bar(equipment_class_counts_ship.index, equipment_class_counts_ship)
        # Set font properties for title
        hfont = {'fontname': 'serif', 'weight': 'bold'}
        # ax.set_title('Equipment Class', size=9, **hfont)
        ax.set_xlabel('Equipment Classes', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_xticklabels(equipment_class_counts_ship.index, rotation=90, fontsize=8)

        # Add percentages on top of the bars
        for i, count in enumerate(equipment_class_counts_ship):
            ax.text(i, count + 0.1, f'{count / len(df) * 100:.1f}%', ha='center', fontsize=7, **hfont)

        # Display the plot in Streamlit
        st.pyplot(fig)




    incsh3, incsh4, incsh5 = st.columns((3))

    with incsh3:
        customer_class_counts_ship = df2["customer_class"].value_counts()
        # Create a new figure and axis
        fig, ax = plt.subplots()
        # Create a pie chart
        ax.pie(customer_class_counts_ship, labels=customer_class_counts_ship.index, autopct='%1.1f%%')
        # Set font properties for title
        hfont = {'fontname': 'serif', 'weight': 'bold'}
        ax.set_title('Customer Class Shipped', size=9, **hfont)
        # Display the plot in Streamlit
        st.pyplot(fig)

    with incsh4:
        # Gruplama işlemini yapın
        customer_class_distr_ship = df2.groupby("customer_class").agg({"total_amount": "sum"})
        customer_class_distr_ship["Ratio"] = customer_class_distr_ship["total_amount"] / customer_class_distr_ship["total_amount"].sum() * 100

        # Streamlit uygulaması
        st.write("<h1 style='font-size: 20px;'>Total Amount by Customer Class Shipped </h1>",
                 unsafe_allow_html=True)
        # Grafiği çizin
        st.bar_chart(customer_class_distr_ship["total_amount"])

    with incsh5 :
        st.write("")

        cust_clas_private =customer_class_distr_ship.iloc[1, 0]
        formatted_cust_clas_private = "{:,.2f}".format(cust_clas_private)
        cust_clas_puplic = customer_class_distr_ship.iloc[0, 0]
        formatted_cust_clas_puplic= "{:,.2f}".format(cust_clas_puplic)


        st.markdown("<h1 style='font-size: 18px;color: black;'> (O)Private Companies Total Amount : {}</h1>".format(
            formatted_cust_clas_private),unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 18px;color: black;'> (K)Public Body Total Amount : {}</h1>".format(formatted_cust_clas_puplic), unsafe_allow_html=True)


    st.write("     ")

    st.write("<h1 style='font-size: 20px;color: black;'> Shipped Customer List </h1>", unsafe_allow_html=True)
    st.write(df2.sort_values("registration_close_date"), ignore_index=True)


    st.write("   ")

    st.markdown("<div style='text-align: center;'><h1 style='font-size: 28px;color: navy;'>-------------    KPIs (Key Performance Indicators)    -------------</h1></div>",
        unsafe_allow_html=True)

    st.write("   ")
    st.write("<h1 style='font-size: 20px;color: navy;'> ------  Indicators of bid preparation and repair times  ------ </h1>", unsafe_allow_html=True)

    time1, time2, time3 = st.columns((3))

    with time1 :
        avg_bid = (df2["offer_date"] - df2["registration_date"]).median()
        avg_bid = str(avg_bid)[:6]
        st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Average Bidding Time  (days)   : {}</h1>".format(
            avg_bid), unsafe_allow_html=True)
    with time2 :
        avg_eq_wsh =(df2["registration_close_date"] - df2["registration_date"]).median()
        avg_eq_wsh = str(avg_eq_wsh)[:6]
        st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Average Time the Equipment is in the Workshop (days)     : {}</h1>".format(
            avg_eq_wsh), unsafe_allow_html=True)
    with time3 :
        avg_eq_repair = (df2["registration_close_date"] - df2["offer_approval_date"]).median()
        avg_eq_repair = str(avg_eq_repair)[:6]
        st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Average Repair Time of equipment (days)   : {}</h1>".format(
            avg_eq_repair), unsafe_allow_html=True)
    st.write("")
    st.write("<h1 style='font-size: 20px;color: navy;'>------  According to equipment classes  ------ </h1>",
             unsafe_allow_html=True)

    selectbox1, selectbox2, selectbox3, selectbox4, selectbox5 = st.columns((5))

    with selectbox1:
        st.write("<h1 style='font-size: 20px;color: navy;'> Select equipment class  :  </h1>",
             unsafe_allow_html=True)
    with selectbox2:
        select_eq_class = st.selectbox('', df2["equipment_class"].unique())


    time4, time5, time6 = st.columns((3))

    with time4:
        avg_bid_class_df = df2[df2["equipment_class"] == select_eq_class]
        avg_bid_class = (avg_bid_class_df["offer_date"] - avg_bid_class_df["registration_date"]).median()
        avg_bid_class = str(avg_bid_class)[:6]
        st.markdown(
        "<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'>  Average Bidding Time according to equipment classes (days) : {}</h1>".format(
            avg_bid_class), unsafe_allow_html=True)
    with time5:
        avg_eq_wsh_class = (avg_bid_class_df["registration_close_date"] - avg_bid_class_df["registration_date"]).median()
        avg_eq_wsh_class = str(avg_eq_wsh_class)[:6]
        st.markdown(
        "<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Average time the equipment is in the workshop according to equipment classes (days) : {}</h1>".format(
            avg_eq_wsh_class), unsafe_allow_html=True)
    with time6 :
        avg_eq_repair_class = (avg_bid_class_df["registration_close_date"] - avg_bid_class_df["offer_approval_date"]).median()
        avg_eq_repair_class = str(avg_eq_repair_class)[:6]
        st.markdown(
        "<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Average Repair Time of equipment according to equipment classes (days)  : {}</h1>".format(
            avg_eq_repair_class), unsafe_allow_html=True)

    st.write("")

    st.write("<h1 style='font-size: 20px;color: navy;'>------  Indicators by price  ------ </h1>",
             unsafe_allow_html=True)

    totalamount1, totalamount2, totalamount3 = st.columns((3))

    with totalamount1:
        total_amount = df2["total_amount"].sum()
        formatted_total_amount = "{:,.2f}".format(total_amount)
        st.markdown( "<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Total Amount ( SEK )   :  {}</h1>".format(
                formatted_total_amount), unsafe_allow_html=True)
        #st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Total Amount ( SEK )   :  {:.2f}</h1>".format(
        #   formatted_total_amount), unsafe_allow_html=True)
    with totalamount2:
        total_amount_sp = (df2["spare_parts_amount"] + df2["oil_coolingfluid_amount"]).sum()
        formatted_total_amount_sp = "{:,.2f}".format(total_amount_sp)
        st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Total Amount Spare Parts ( SEK ) :  {}</h1>".format(
            formatted_total_amount_sp), unsafe_allow_html=True)

    with totalamount3:
        total_amount_excl_sp = (df2["labor_amount"] + df2["stator_winding_amount"] + df2["shaft_repair_amount"] + df2["welding_and_turning_amount"]).sum()
        formatted_total_amount_excl_sp = "{:,.2f}".format(total_amount_excl_sp)
        st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Total Amount Excluding Spare Parts ( SEK ) :  {}</h1>".format(
            formatted_total_amount_excl_sp), unsafe_allow_html=True)

    ratiocol1, ratiocol2, ratiocol3, ratiocol4 = st.columns((4))
    with ratiocol1:
        sp_ratio = total_amount_sp / total_amount * 100
        st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Total Spare Parts Ratio % :  {:.2f}</h1>".format(
            sp_ratio), unsafe_allow_html=True)
    with ratiocol2:
        lab_ratio = total_amount_excl_sp / total_amount * 100
        st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Total Labor Amount Ratio %  :  {:.2f}</h1>".format(
            lab_ratio), unsafe_allow_html=True)



    st.write("")

    st.write("")

    st.write("<h1 style='font-size: 20px;color: black;'> Total Amount by Customers ( SEK ) </h1>", unsafe_allow_html=True)

    st.write(df2.groupby("customer_name").agg({"spare_parts_amount": "sum",
                                             "labor_amount": "sum",
                                             "stator_winding_amount": "sum",
                                             "shaft_repair_amount": "sum",
                                             "welding_and_turning_amount": "sum",
                                             "oil_coolingfluid_amount": "sum",
                                             "total_amount": "sum"}).sort_values("total_amount", ascending=False))

    st.write("")

    st.write("<h1 style='font-size: 20px;color: black;'> Total Number of Shipped Equipment :  </h1>",
             unsafe_allow_html=True)
    st.write(df2.groupby("customer_name").agg({"service_request_number": "count"}).sort_values("service_request_number", ascending=False))


def data_pred():
    st.header('CLTV Prediction')




    def af_price_data_prep(dataframe):
        dataframe["customer_class"] = dataframe["customer_id"].str[0]

        def grab_col_names(dataframe, cat_th=10, car_th=20):
            """
            Returns the names of categorical, numeric and categorical but cardinal variables in the data set.

            Parameters
            ----------
            dataframe: dataframe
                variable names are the dataframe to be retrieved.
            cat_th: int, float
                class threshold for numeric but categorical variables
            car_th: int, float
                class threshold for categorical but cardinal variables

            Returns
            -------
            cat_cols: list
                Categorical variables list
            num_cols: list
                Numeric variable list
            cat_but_car: list
                List of cardinal variables with categorical view

            Notes
            ------
            cat_cols + num_cols + cat_but_car = total number of variables
            num_but_cat is inside cat_cols.

            """
            # cat_cols, cat_but_car
            cat_cols = [col for col in dataframe.columns if
                        str(dataframe[col].dtypes) in ["category", "object", "bool"]]

            num_but_cat = [col for col in dataframe.columns if
                           dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

            cat_but_car = [col for col in dataframe.columns if
                           dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

            cat_cols = cat_cols + num_but_cat
            cat_cols = [col for col in cat_cols if col not in cat_but_car]

            num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
            num_cols = [col for col in num_cols if col not in cat_cols]

            print(f"Observations: {dataframe.shape[0]}")
            print(f"Variables: {dataframe.shape[1]}")
            print(f'cat_cols: {len(cat_cols)}')
            print(f'num_cols: {len(num_cols)}')
            print(f'cat_but_car: {len(cat_but_car)}')
            print(f'num_but_cat: {len(num_but_cat)}')

            return cat_cols, num_cols, cat_but_car, num_but_cat

        cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)

        cat_cols = cat_cols + cat_but_car

        df = dataframe

        print(df.isnull().sum())
        print(df.info())
        print("################################")
        print(df.head())

        return df

    df3 = af_price_data_prep(data)


    kcol1, kcol2 = st.columns((2))
    startDate = pd.to_datetime(df3["registration_close_date"]).min()
    endDate = pd.to_datetime(df3["registration_close_date"]).max()

    with kcol1:
        date1 = pd.to_datetime(st.date_input(" Start Date  ", startDate))

    with kcol2:
        date2 = pd.to_datetime(st.date_input(" End Date  ", endDate))

    df3 = df3.loc[(df3["registration_close_date"] >= date1) & (df3["registration_close_date"] <= date2)].copy()

    month_ = st.selectbox('Select month', [None, 3, 6, 9, 12])

    analysis_date = date2 + pd.Timedelta(days=1)
    formatted_date = analysis_date.strftime('%Y-%m-%d')
    st.write('<span style="font-size:20px">Analysis date  : </span>', formatted_date, unsafe_allow_html=True)

    if st.button("Run"):


        def create_cltv_p(dataframe, month=month_):

            # 1. Veri Ön İşleme

            df_train = df3.loc[(df3["registration_close_date"] >= date1) & (df3["registration_close_date"] <= date2)]


            cltv_df = df_train.groupby("customer_name").agg(
                {"registration_close_date": [lambda x: (analysis_date - x.min()).days,
                                          lambda x: (x.max() - x.min()).days],
                 "service_request_number": "count",
                 "total_amount": lambda x: x.sum()
                 })

            cltv_df.columns = cltv_df.columns.droplevel(0)
            cltv_df.columns = ['T', 'recency', 'frequency', 'monetary']
            # monetary: satın alma başına ortalama kazanç
            cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
            # frequency: tekrar eden toplam satın alma sayısı (frequency>1)
            cltv_df = cltv_df[cltv_df["frequency"] > 1]

            cltv_df["recency"] = cltv_df["recency"] / 7
            cltv_df["T"] = cltv_df["T"] / 7
            cltv_df = cltv_df[cltv_df["recency"] > 0]

            # 2. BG-NBD Modelinin Kurulması
            bgf = BetaGeoFitter(penalizer_coef=0.001)
            bgf.fit(cltv_df['frequency'],
                    cltv_df['recency'],
                    cltv_df['T'])

            cltv_df[f"expected_purc_{month_}_month"] = bgf.predict(4 * month_,
                                                                   cltv_df['frequency'],
                                                                   cltv_df['recency'],
                                                                   cltv_df['T'])

            # 3. GAMMA-GAMMA Modelinin Kurulması
            ggf = GammaGammaFitter(penalizer_coef=0.01)
            ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
            cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                         cltv_df['monetary'])

            # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
            cltv = ggf.customer_lifetime_value(bgf,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'],
                                               cltv_df['monetary'],
                                               time=month_,  # aylık
                                               freq="W",  # T'nin frekans bilgisi.
                                               discount_rate=0.01)

            cltv = cltv.reset_index()
            cltv_final = cltv_df.merge(cltv, on="customer_name", how="left")
            cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])


            return cltv_final

        cltv_final = create_cltv_p(df3, month=month_)
        st.write(cltv_final.sort_values(by="clv", ascending=False))

        total_cltv = cltv_final["clv"].sum()
        formatted_total_cltv = "{:,.2f}".format(total_cltv)

        st.markdown(
            "<h1 style='text-align: left; color: navy; font-size: 20px; font-weight: bold;'>Total of CLTV     : {}</h1>".format(
                formatted_total_cltv), unsafe_allow_html=True)


        st.markdown(
            "<h1 style='text-align: left; color: navy; font-size: 20px; font-weight: bold;'>Total of Expected Purchase     : {}</h1>".format(
                round(cltv_final[f"expected_purc_{month_}_month"].sum())), unsafe_allow_html=True)


        segment_mean_table = cltv_final.groupby("segment").agg({"clv":"mean",
                                           "T":"mean",
                                           "recency":"mean",
                                           "frequency":"mean",
                                           "monetary":"mean"})



        st.write("<h1 style='font-size: 20px;color: navy;'>According to Classes ( mean ) </h1>",
                 unsafe_allow_html=True)
        st.write(segment_mean_table)

    else:
        st.write(f"Sorry, '{month_}' not found in the dictionary.")


def rfm_analysis():
    st.write("")


    st.header('RFM Analysis')

    def af_price_data_prep(dataframe):
        dataframe["customer_class"] = dataframe["customer_id"].str[0]

        def grab_col_names(dataframe, cat_th=10, car_th=20):
            """
            Returns the names of categorical, numeric and categorical but cardinal variables in the data set.

            Parameters
            ----------
            dataframe: dataframe
                variable names are the dataframe to be retrieved.
            cat_th: int, float
                class threshold for numeric but categorical variables
            car_th: int, float
                class threshold for categorical but cardinal variables

            Returns
            -------
            cat_cols: list
                Categorical variables list
            num_cols: list
                Numeric variable list
            cat_but_car: list
                List of cardinal variables with categorical view

            Notes
            ------
            cat_cols + num_cols + cat_but_car = total number of variables
            num_but_cat is inside cat_cols.

            """
            # cat_cols, cat_but_car
            cat_cols = [col for col in dataframe.columns if
                        str(dataframe[col].dtypes) in ["category", "object", "bool"]]

            num_but_cat = [col for col in dataframe.columns if
                           dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

            cat_but_car = [col for col in dataframe.columns if
                           dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

            cat_cols = cat_cols + num_but_cat
            cat_cols = [col for col in cat_cols if col not in cat_but_car]

            num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
            num_cols = [col for col in num_cols if col not in cat_cols]

            print(f"Observations: {dataframe.shape[0]}")
            print(f"Variables: {dataframe.shape[1]}")
            print(f'cat_cols: {len(cat_cols)}')
            print(f'num_cols: {len(num_cols)}')
            print(f'cat_but_car: {len(cat_but_car)}')
            print(f'num_but_cat: {len(num_but_cat)}')

            return cat_cols, num_cols, cat_but_car, num_but_cat

        cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)

        cat_cols = cat_cols + cat_but_car

        df = dataframe

        print(df.isnull().sum())
        print(df.info())
        print("################################")
        print(df.head())

        return df

    df4 = af_price_data_prep(data)


    kcol1, kcol2 = st.columns((2))
    startDate = pd.to_datetime(df4["registration_close_date"]).min()
    endDate = pd.to_datetime(df4["registration_close_date"]).max()

    with kcol1:
        date1 = pd.to_datetime(st.date_input(" Start Date  ", startDate))

    with kcol2:
        date2 = pd.to_datetime(st.date_input(" End Date  ", endDate))

    df4 = df4.loc[(df4["registration_close_date"] >= date1) & (df4["registration_close_date"] <= date2)].copy()


    analysis_date = date2 + pd.Timedelta(days=1)
    formatted_date = analysis_date.strftime('%Y-%m-%d')
    st.write('<span style="font-size:20px">Analysis date  : </span>', formatted_date, unsafe_allow_html=True)


    df_train = df4.loc[(df4["registration_close_date"] >= date1) & (df4["registration_close_date"] <= date2)]


    def af_rfm_prep(dataframe):
        rfm = dataframe.groupby("customer_name").agg({"registration_close_date": lambda x: (analysis_date - x.max()).days,
                                                    "service_request_number": lambda x: x.nunique(),
                                                    "total_amount": lambda x: x.sum()})

        rfm.columns = ["recency", "frequency", "monetary"]

        rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
        rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

        # RFM Skor
        rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

        seg_map = {
            r'[1-2][1-2]': 'hibernating',
            r'[1-2][3-4]': 'at_Risk',
            r'[1-2]5': 'cant_loose',
            r'3[1-2]': 'about_to_sleep',
            r'33': 'need_attention',
            r'[3-4][4-5]': 'loyal_customers',
            r'41': 'promising',
            r'51': 'new_customers',
            r'[4-5][2-3]': 'potential_loyalists',
            r'5[4-5]': 'champions'
                }
        rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)

        return rfm

    rfm = af_rfm_prep(df_train)

    st.write("")

    st.write("")



    st.markdown("<h1 style='font-size: 30px;'>Number of Customers by Segments</h1>", unsafe_allow_html=True)

    rfm_coordinates = {"champions": [3, 5, 0.8, 1],
                       "loyal_customers": [3, 5, 0.4, 0.8],
                       "cant_loose": [4, 5, 0, 0.4],
                       "at_Risk": [2, 4, 0, 0.4],
                       "hibernating": [0, 2, 0, 0.4],
                       "about_to_sleep": [0, 2, 0.4, 0.6],
                       "promising": [0, 1, 0.6, 0.8],
                       "new_customers": [0, 1, 0.8, 1],
                       "potential_loyalists": [1, 3, 0.6, 1],
                       "need_attention": [2, 3, 0.4, 0.6]}

    sns.set_style('whitegrid')
    palette = 'Set2'

    plt.figure(figsize=(18, 8))
    ax = sns.countplot(data=rfm,
                       x='segment',
                       palette=palette)
    total = len(rfm.segment)
    for patch in ax.patches:
        percentage = '{:.1f}%'.format(100 * patch.get_height() / total)
        x = patch.get_x() + patch.get_width() / 2 - 0.17
        y = patch.get_y() + patch.get_height() * 1.005
        ax.annotate(percentage, (x, y), size=14)
    plt.title('Number of Customers by Segments', size=16)
    plt.xlabel('Segment', size=14)
    plt.ylabel('Count', size=14)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.show()
    st.pyplot(plt)

    st.markdown("<h1 style='font-size: 30px;'>Recency & Frequency by Segments</h1>", unsafe_allow_html=True)

    plt.figure(figsize=(18, 8))
    sns.scatterplot(data=rfm,
                    x='recency',
                    y='frequency',
                    hue='segment',
                    palette=palette,
                    s=60)
    plt.title('Recency & Frequency by Segments', size=16)
    plt.xlabel('RECENCY', size=12)
    plt.ylabel('FREQUENCY', size=12)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.legend(loc='best', fontsize=12,
               title='Segments', title_fontsize=14)
    plt.show()
    st.pyplot(plt)


    st.write("")

    st.write("")

    st.markdown("<h1 style='font-size: 30px;'> Treemap </h1>", unsafe_allow_html=True)




    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])

    plt.rcParams["axes.facecolor"] = "white"
    palette = ["#282828", "#04621B", "#971194", "#F1480F", "#4C00FF",
               "#FF007B", "#9736FF", "#8992F3", "#B29800", "#80004C"]

    for key, color in zip(rfm_coordinates.keys(), palette[:10]):
        coordinates = rfm_coordinates[key]
        ymin, ymax, xmin, xmax = coordinates[0], coordinates[1], coordinates[2], coordinates[3]

        ax.axhspan(ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, facecolor=color)

        users = rfm[rfm.segment == key].shape[0]
        users_percentage = (rfm[rfm.segment == key].shape[0] / rfm.shape[0]) * 100
        avg_monetary = rfm[rfm.segment == key]["monetary"].mean()

        user_txt = "\n\nTotal Users: " + str(users) + "(" + str(round(users_percentage, 2)) + "%)"
        monetary_txt = "\n\n\n\nAverage Monetary: " + str(round(avg_monetary, 2))

        x = 5 * (xmin + xmax) / 2
        y = (ymin + ymax) / 2

        plt.text(x=x, y=y, s=key, ha="center", va="center", fontsize=18, color="white", fontweight="bold")
        plt.text(x=x, y=y, s=user_txt, ha="center", va="center", fontsize=14, color="white")
        plt.text(x=x, y=y, s=monetary_txt, ha="center", va="center", fontsize=14, color="white")

        ax.set_xlabel("Recency Score")
        ax.set_ylabel("Frequency Score")

    sns.despine(left=True, bottom=True)
    plt.show()
    st.pyplot(fig)

    st.write("   ")

    st.markdown("<h1 style='font-size: 25px;'> Segment statistics </h1>", unsafe_allow_html=True)

    st.write(rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"]))

    st.write("")

    st.markdown("<h1 style='font-size: 30px;'> Segment overview </h1>", unsafe_allow_html=True)

    segments =["champions", "loyal_customers", "cant_loose", "at_Risk", "hibernating","about_to_sleep",
               "promising", "new_customers", "potential_loyalists","need_attention"]

    select_segment = st.selectbox('Select segment',segments)

    seg_index = rfm[rfm["segment"] == select_segment].index

    df_seg =df4[(df4["customer_name"].isin(seg_index))]

    st.write(df_seg.groupby("customer_name").agg({"total_amount": "sum"}).sort_values("total_amount", ascending=False))

    st.write("")

    st.markdown("<h1 style='font-size: 30px;'> Customer overview </h1>", unsafe_allow_html=True)

    customers =df4["customer_name"].unique()

    select_customers = st.selectbox('Select customer', customers)

    st.write(df4[df4["customer_name"] == select_customers])

def cohort_analysis():
    st.header('Cohort Analysis')

    def af_price_data_prep(dataframe):
        dataframe["customer_class"] = dataframe["customer_id"].str[0]

        def grab_col_names(dataframe, cat_th=10, car_th=20):
            """
            Returns the names of categorical, numeric and categorical but cardinal variables in the data set.

            Parameters
            ----------
            dataframe: dataframe
                variable names are the dataframe to be retrieved.
            cat_th: int, float
                class threshold for numeric but categorical variables
            car_th: int, float
                class threshold for categorical but cardinal variables

            Returns
            -------
            cat_cols: list
                Categorical variables list
            num_cols: list
                Numeric variable list
            cat_but_car: list
                List of cardinal variables with categorical view

            Notes
            ------
            cat_cols + num_cols + cat_but_car = total number of variables
            num_but_cat is inside cat_cols.

            """
            # cat_cols, cat_but_car
            cat_cols = [col for col in dataframe.columns if
                        str(dataframe[col].dtypes) in ["category", "object", "bool"]]

            num_but_cat = [col for col in dataframe.columns if
                           dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

            cat_but_car = [col for col in dataframe.columns if
                           dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

            cat_cols = cat_cols + num_but_cat
            cat_cols = [col for col in cat_cols if col not in cat_but_car]

            num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
            num_cols = [col for col in num_cols if col not in cat_cols]

            print(f"Observations: {dataframe.shape[0]}")
            print(f"Variables: {dataframe.shape[1]}")
            print(f'cat_cols: {len(cat_cols)}')
            print(f'num_cols: {len(num_cols)}')
            print(f'cat_but_car: {len(cat_but_car)}')
            print(f'num_but_cat: {len(num_but_cat)}')

            return cat_cols, num_cols, cat_but_car, num_but_cat

        cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)

        cat_cols = cat_cols + cat_but_car

        df = dataframe

        print(df.isnull().sum())
        print(df.info())
        print("################################")
        print(df.head())

        return df

    df5 = af_price_data_prep(data)

    kcol1, kcol2 = st.columns((2))
    startDate = pd.to_datetime(df5["registration_close_date"]).min()
    endDate = pd.to_datetime(df5["registration_close_date"]).max()

    with kcol1:
        date1 = pd.to_datetime(st.date_input(" Start Date  ", startDate))

    with kcol2:
        date2 = pd.to_datetime(st.date_input(" End Date  ", endDate))

    df5 = df5.loc[(df5["registration_close_date"] >= date1) & (df5["registration_close_date"] <= date2)].copy()

    if st.button("Run"):

        def cohort_func(dataframe):
            df_cohort = dataframe


            def take_months(x):
                return dt.datetime(x.year, x.month, 1)

            df_cohort['invoicemonth'] = df_cohort['registration_close_date'].apply(take_months)
            df_cohort['cohortmonth'] = df_cohort.groupby('customer_name')['invoicemonth'].transform('min')

            def get_month_int(dframe, column):
                year = dframe[column].dt.year
                month = dframe[column].dt.month
                day = dframe[column].dt.day
                return year, month, day

            invoice_year, invoice_month, invoice_day = get_month_int(df_cohort, 'invoicemonth')
            cohort_year, cohort_month, cohort_day = get_month_int(df_cohort, 'cohortmonth')

            year_diff = invoice_year - cohort_year
            month_diff = invoice_month - cohort_month

            df_cohort['CohortIndex'] = year_diff * 12 + month_diff + 1

            cohort1 = df_cohort.groupby(['cohortmonth', 'CohortIndex'])['customer_name'].nunique().reset_index()

            pivot_cohort1 = cohort1.reset_index().pivot(index='cohortmonth', columns='CohortIndex',
                                                    values='customer_name').round(1)

            sizes = pivot_cohort1.iloc[:, 0]
            retention = pivot_cohort1.divide(sizes, axis=0).round(3)  # axis=0 to ensure the divide along the row axis

            return retention

        retention = cohort_func(df5)

        st.write("")

        st.write("")

        plt.figure(figsize=(16, 9))
        plt.title('Retention rates')
        sns.heatmap(data=retention, annot=True, fmt='.0%', vmin=0.0, vmax=0.5, cmap="GnBu")
        plt.show()
        st.pyplot(plt)






# Add a title and intro text
st.title('Technical Service Department Dataset')
st.text('This is a web app to Technical Service Department Data')

# Sidebar setup
st.sidebar.title('Sidebar')
fl = st.sidebar.file_uploader(":file_folder: Upload an Excel file", type=(["xlsx", "xls"]))

if fl is not None:
    filename = fl.name
    st.write(filename)
    data = pd.read_excel(fl, engine="openpyxl", dtype={'SERİ_NO': str})
    # Continue with processing the data or displaying it as needed
else:
    st.warning("Please upload an Excel file.")


#Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', ['Home', 'Reporting & Analysis', 'RFM Analysis','CLTV Prediction', 'Cohort Analysis'])


# Navigation options
if options == 'Home':
    home(fl)
elif options == 'Reporting & Analysis':
    data_summary()
elif options == 'CLTV Prediction':
    data_pred()

elif options == 'RFM Analysis':
    rfm_analysis()
elif options == 'Cohort Analysis':
    cohort_analysis()

date_col = ["registration_date", "offer_date", "offer_approval_date", "registration_close_date"]
# Loop through the columns and apply date formatting
for col in date_col:
    data[col] = pd.to_datetime(data[col]).dt.strftime('%Y-%m-%d')