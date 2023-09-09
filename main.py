from googlesearch import search
import os
import time
from bs4 import BeautifulSoup
import re
import requests
import warnings
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import langchain
from langchain import PromptTemplate 
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI 
from langchain.agents import load_tools, AgentType, Tool, initialize_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain 

# Streamlit 
import streamlit as st 

OPENAI_API_KEY = os.environ.get('OPEN_API_KEY') 
warnings.filterwarnings("ignore")

# load LLM funciton 
def load_llm():  
    llm = ChatOpenAI(temperature=.25, openai_api_key=OPENAI_API_KEY )  
    return llm 

# check openAI key 
def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key (or set it as .env variable)",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

# Coonect funciton for Coinbase public API 
def connect(url, *args, **kwargs):
    try:
        if kwargs.get('param', None) is not None:
            params = kwargs.get('param')
            response = requests.get(url,params)
        else:
            response = requests.get(url)
        # print if an error occurs 
        response.raise_for_status()
        print(f'HTTP connection {url} successful!')
        return response
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')

##### PLOT setting 
CANDLE_INCREASE_COLOR = 'cyan'
CANDLE_DECREASE_COLOR = 'gray'
MACD_COLOR = '#A9B1F5'
MACD_SIGNAL_COLOR = '#FFD1A3'
RED_COLOR = '#D61170'
GREEN_COLOR = '#3FFA68'
#PLOTLY_TEMPLATE = 'plotly_dark'
PLOTLY_TEMPLATE = 'plotly'


# Start streamlit 
st.set_page_config(page_title="Crypto Analysis", page_icon="🤑")

# Start top information 
st.header("GPT Crypto Analysis 🤑")

st.markdown("You want to save time analyze a cryptocurrency? This tool is meant to help you decide whether to buy a crypto or not.\
            \n\nThis tool is powered by [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/#), [markdownify](https://pypi.org/project/markdownify/), [LangChain](https://langchain.com/), and [OpenAI](https://openai.com)" \
            )

# Collect information about the person you want to research
currency = st.text_input(label="currency",  placeholder="Ex: ETH", key="currency")

# Choose currency that should be used to display data
FIAT_CURRENCIES = ['USD']
MY_QUOTE_CURRENCY = FIAT_CURRENCIES[0]
GRANULARITIES =  ['DAILY','60MIN','15MIN','1MIN']

# Convenience method that establishes a connection to Coinbase and will retrieve historic data for the chosen interval
def get_historic_data(start_date, end_date, interval, base_currency, quote_currency):

    VALID_INTERVAL = {'DAILY', '60MIN', '15MIN', '1MIN'}

    if interval not in VALID_INTERVAL:
        raise ValueError("results: interval must be one of %r." % VALID_INTERVAL)

    if interval == '1MIN':
        granularity = '60'
    elif interval == '15MIN':
        granularity = '900'
    elif interval == '60MIN':
        granularity = '3600'
    else: # DAILY as the default
        granularity = '86400'

    params = {'start':start_date, 'end':end_date, 'granularity':granularity}
    data = json.loads(connect('https://api.exchange.coinbase.com/products/'+quote_currency+'-'+base_currency+'/candles', param = params).text)
    for x in data:
        x.append(quote_currency)
        x.append(interval)
    return data

# Fetch crypto data from Coinbase 
def get_crypto_price(currency):
    ''' 
      input: 
        - currency: ['BTC']
      output:
        - DF of crypto info including 
            - current val
            - 1 min trend : Bear, BUll 
            - 1 min prev trend:  Bear, BUll 
            - 15 min trend : Bear, BUll 
            - 15 min prev trend:  Bear, BUll 
            - 60 min trend : Bear, BUll 
            - 60 min prev trend:  Bear, BUll 
            - 24_low: 
            - 30day_low:
            - 90day_low: 
            - 24_high: 
            - 30day_high:
            - 90day_high:
            - 90day_max_high_low_span: %
            - 12day_max_high_low_span: %

    '''
    # Avoid rate limit error
    # time.sleep(4) 

    AVERAGE_TRUE_RANGE_PERIODS = 14
    currency_history_rows_enhanced = []
    
    MY_CRYPTO_CURRENCIES = [currency] # a list  
    

    # Get time
    server_time = json.loads(connect('https://api.exchange.coinbase.com/time').text)
    # Server time does not comply to iso format, therefore slight modification of string needed
    server_time_now = datetime.fromisoformat(server_time['iso'].replace('T', ' ', 1)[0:19]) 

    ############ get history data ###############
    currency_history_rows = []
    for currency in MY_CRYPTO_CURRENCIES:
        end_date = server_time_now.isoformat()
        # 1 minutes data:
        start_date = (server_time_now - timedelta(hours=2)).isoformat()
        currency_history_rows.extend(get_historic_data(start_date,end_date,'1MIN', MY_QUOTE_CURRENCY, currency))
        # 15 minutes data:
        start_date = (server_time_now - timedelta(hours=75)).isoformat()
        currency_history_rows.extend(get_historic_data(start_date,end_date,'15MIN', MY_QUOTE_CURRENCY, currency))
        # 60 minutes data:
        start_date = (server_time_now - timedelta(hours=300)).isoformat()
        currency_history_rows.extend(get_historic_data(start_date,end_date,'60MIN', MY_QUOTE_CURRENCY, currency))
        # Daily data:
        start_date = (server_time_now - timedelta(days=90)).isoformat()
        currency_history_rows.extend(get_historic_data(start_date,end_date,'DAILY', MY_QUOTE_CURRENCY, currency))
    df_history = pd.DataFrame(currency_history_rows)
    # Add column names in line with the Coinbase documentation
    df_history.columns = ['time','low','high','open','close','volume','base_currency','granularity']
    # We will add a few more columns just for better readability
    df_history['quote_currency'] = MY_QUOTE_CURRENCY
    df_history['date'] = pd.to_datetime(df_history['time'], unit='s')
    df_history['year'] = pd.DatetimeIndex(df_history['date']).year
    df_history['month'] = pd.DatetimeIndex(df_history['date']).month
    df_history['day'] = pd.DatetimeIndex(df_history['date']).day
    df_history['hour'] = pd.DatetimeIndex(df_history['date']).hour
    df_history['minute'] = pd.DatetimeIndex(df_history['date']).minute

    for currency in MY_CRYPTO_CURRENCIES:
        for granularity in GRANULARITIES:
            df_history_currency = df_history.query('granularity == @granularity & base_currency == @currency').copy()
            # Oldest to newest date sorting needed for SMA calculation
            df_history_currency = df_history_currency.sort_values(['date'], ascending=True)
            df_history_currency['SMA3'] = df_history_currency['close'].rolling(window=3).mean()
            df_history_currency['SMA7'] = df_history_currency['close'].rolling(window=7).mean()
            df_history_currency['EMA12'] = df_history_currency['close'].ewm(span=12, adjust=False).mean()
            df_history_currency['EMA26'] = df_history_currency['close'].ewm(span=26, adjust=False).mean()
            df_history_currency['MACD'] = df_history_currency['EMA12'] - df_history_currency['EMA26']
            df_history_currency['MACD_signal'] = df_history_currency['MACD'].ewm(span=9, adjust=False).mean()
            df_history_currency['macd_histogram'] = ((df_history_currency['MACD']-df_history_currency['MACD_signal']))
            df_history_currency['open_to_close_perf'] = ((df_history_currency['close']-df_history_currency['open']) / df_history_currency['open'])
            df_history_currency['high_low_diff'] = (df_history_currency['high']-df_history_currency['low'])
            df_history_currency['high_prev_close_abs'] = np.abs(df_history_currency['high'] - df_history_currency['close'].shift())
            df_history_currency['low_prev_close_abs'] = np.abs(df_history_currency['low'] - df_history_currency['close'].shift())
            df_history_currency['true_range'] = df_history_currency[['high_low_diff', 'high_prev_close_abs', 'low_prev_close_abs']].max(axis=1)
            #df_history_currency['average_true_range'] = df_history_currency['true_range'].rolling(window=14).sum()/14
            df_history_currency['average_true_range'] = df_history_currency['true_range'].ewm(alpha=1/AVERAGE_TRUE_RANGE_PERIODS, min_periods=AVERAGE_TRUE_RANGE_PERIODS, adjust=False).mean()
            df_history_currency['high_low_perf'] = ((df_history_currency['high']-df_history_currency['low']) / df_history_currency['high'])
            df_history_currency['open_perf_last_3_period_abs'] = df_history_currency['open'].rolling(window=4).apply(lambda x: x.iloc[1] - x.iloc[0])
            df_history_currency['open_perf_last_3_period_per'] = df_history_currency['open'].rolling(window=4).apply(lambda x: (x.iloc[1] - x.iloc[0])/x.iloc[0])
            df_history_currency['bull_bear'] = np.where(df_history_currency['macd_histogram']< 0, 'Bear', 'Bull')
            currency_history_rows_enhanced.append(df_history_currency)
    df_history_enhanced = pd.concat(currency_history_rows_enhanced, ignore_index=True)
    # Last step to tag changes in market trends from one period to the other (sorting important)
    df_history_enhanced = df_history_enhanced.sort_values(['base_currency','granularity','date'], ascending=True)
    df_history_enhanced['market_trend_continued'] = df_history_enhanced.bull_bear.eq(df_history_enhanced.bull_bear.shift()) & df_history_enhanced.base_currency.eq(df_history_enhanced.base_currency.shift()) & df_history_enhanced.granularity.eq(df_history_enhanced.granularity.shift())

    ############### 24hr statistic ##################
    currency_rows = []
    for currency in MY_CRYPTO_CURRENCIES:
        data = json.loads(connect('https://api.exchange.coinbase.com/products/'+currency+'-'+MY_QUOTE_CURRENCY+'/stats').text)
        currency_rows.append(data)
    df_24hstats = pd.DataFrame(currency_rows, index = MY_CRYPTO_CURRENCIES)
    df_24hstats['base_currency'] = df_24hstats.index
    df_24hstats['quote_currency'] = MY_QUOTE_CURRENCY
    df_24hstats['open'] = df_24hstats['open'].astype(float)
    df_24hstats['high'] = df_24hstats['high'].astype(float)
    df_24hstats['low'] = df_24hstats['low'].astype(float)
    df_24hstats['volume'] = df_24hstats['volume'].astype(float)
    df_24hstats['last'] = df_24hstats['last'].astype(float)
    df_24hstats['volume_30day'] = df_24hstats['volume_30day'].astype(float)
    df_24hstats['performance'] = ((df_24hstats['last']-df_24hstats['open']) / df_24hstats['open']) * 100
    df_24hstats_formatted = df_24hstats.copy()
    df_24hstats_formatted['performance'] = df_24hstats_formatted['performance'].apply(lambda x: "{:.2f}%".format((x)))
    df_24hstats_formatted['open'] = df_24hstats_formatted['open'].apply(lambda x: "${:,.2f}".format((x)))
    df_24hstats_formatted['high'] = df_24hstats_formatted['high'].apply(lambda x: "${:,.2f}".format((x)))
    df_24hstats_formatted['low'] = df_24hstats_formatted['low'].apply(lambda x: "${:,.2f}".format((x)))
    df_24hstats_formatted['last'] = df_24hstats_formatted['last'].apply(lambda x: "${:,.2f}".format((x)))

    ############## get additional info DF ##############
    additional_info_rows = []
    for currency in MY_CRYPTO_CURRENCIES:
        df_history_enhanced = df_history_enhanced.sort_values(['date'], ascending=True) #oldest to newest
        row_data = [currency,
            float(df_24hstats.query('base_currency == @currency')['last'].to_string(index=False)), # current value
            df_history_enhanced.query('granularity == \'1MIN\' and base_currency == @currency').tail(1)['bull_bear'].to_string(index=False), # 1min market trend
            df_history_enhanced.query('granularity == \'1MIN\' and base_currency == @currency').iloc[-2]['bull_bear'], # prev 1min market trend
            df_history_enhanced.query('granularity == \'15MIN\' and base_currency == @currency').tail(1)['bull_bear'].to_string(index=False), #15min market trend
            df_history_enhanced.query('granularity == \'15MIN\' and base_currency == @currency').iloc[-2]['bull_bear'], # prev 15min market trend
            df_history_enhanced.query('granularity == \'60MIN\' and base_currency == @currency').tail(1)['bull_bear'].to_string(index=False), # 60min market trend
            df_history_enhanced.query('granularity == \'60MIN\' and base_currency == @currency').iloc[-2]['bull_bear'], # prev 60min market trend
            df_history_enhanced.query('granularity == \'60MIN\' and base_currency == @currency').tail(24)['low'].min(), # lowest last 24h
            df_history_enhanced.query('granularity == \'DAILY\' and base_currency == @currency').tail(30)['low'].min(), # lowest last 30d
            df_history_enhanced.query('granularity == \'DAILY\' and base_currency == @currency').tail(90)['low'].min(), # lowest last 90d
            df_history_enhanced.query('granularity == \'60MIN\' and base_currency == @currency').tail(24)['high'].max(), # highest last 24h
            df_history_enhanced.query('granularity == \'DAILY\' and base_currency == @currency').tail(30)['high'].max(), # highest last 30d
            df_history_enhanced.query('granularity == \'DAILY\' and base_currency == @currency').tail(90)['high'].max(), # highest last 90d
            round(df_history_enhanced.query('granularity == \'DAILY\' and base_currency == @currency').tail(90)['high_low_perf'].max(),3), # max daily high/low difference last 90d
            round(df_history_enhanced.query('granularity == \'60MIN\' and base_currency == @currency')['high_low_perf'].max(),3) # max daily high/low difference last 12.5d
            ]
        additional_info_rows.append(row_data)

    df_additional_info = pd.DataFrame(additional_info_rows, columns = ['base_currency',
                                                                        'current_val',
                                                                        '1min_trend',
                                                                        '1min_prev_trend',
                                                                        '15min_trend',
                                                                        '15min_prev_trend',
                                                                        '60min_trend',
                                                                        '60min_prev_trend',
                                                                        '24h_low',
                                                                        '30day_low',
                                                                        '90day_low',
                                                                        '24h_high',
                                                                        '30day_high',
                                                                        '90day_high',
                                                                        '90day_max_high_low_span',
                                                                        '12day_max_high_low_span'])
    df_additional_info['current_val'] = df_additional_info['current_val'].apply(lambda x: "${:,.2f}".format((x)))
    df_additional_info['24h_low'] = df_additional_info['24h_low'].apply(lambda x: "${:,.2f}".format((x)))
    df_additional_info['30day_low'] = df_additional_info['30day_low'].apply(lambda x: "${:,.2f}".format((x)))
    df_additional_info['90day_low'] = df_additional_info['90day_low'].apply(lambda x: "${:,.2f}".format((x)))
    df_additional_info['24h_high'] = df_additional_info['24h_high'].apply(lambda x: "${:,.2f}".format((x)))
    df_additional_info['30day_high'] = df_additional_info['30day_high'].apply(lambda x: "${:,.2f}".format((x)))
    df_additional_info['90day_high'] = df_additional_info['90day_high'].apply(lambda x: "${:,.2f}".format((x)))
    df_additional_info['90day_max_high_low_span'] = (df_additional_info['90day_max_high_low_span']*100).apply(lambda x: "{:.2f}%".format((x)))
    df_additional_info['12day_max_high_low_span'] = (df_additional_info['12day_max_high_low_span']*100).apply(lambda x: "{:.2f}%".format((x)))
    df_additional_info.set_index('base_currency').transpose()
    
    return df_history_enhanced, df_additional_info

# Script to scrap top5 google news for given company name
def get_recent_crypto_news(currency):
    # time.sleep(4) #To avoid rate limit error
    texts = []
    search_term = currency + ' news'
    # get links
    links = search(search_term, num_results=5, lang='en')
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

    # Get content into a texts list
    for link in links:
      #print(f"{result}")
      page = requests.get(link, headers=headers).text
      soup = BeautifulSoup(page, 'html.parser')
      
      # remove all script and style elements
      for script in soup(["script", "style"]):
          script.extract()   
      
      page_text = soup.getText()
      #print(page_text)
      
      # break into lines and remove leading and trailing space on each
      lines = (line.strip() for line in page_text.splitlines())
      
      # break multi-headlines into a line each
      chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
      
      # drop blank lines
      page_text = ' '.join(chunk for chunk in chunks if chunk)

      #print(page_text)
      page_text = page_text.replace(u'\xa0', ' ')
      
      texts.append(page_text)
      pass

    # make list into string 
    news = "Top 5 recent news:\n"
    for i,n in enumerate(texts):
        news += f"News number {i}: {n}\n"
  
    return news

# Function to change long text about a topic into document 
def split_news(info):
    # make splitter 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=50)

    # split info into docs 
    spt_news  = text_splitter.create_documents([info])

    return spt_news 

def get_crypto_str(df):
    df = df.T
    df_str = df.to_string(index=True, header=False, col_space=None).split('\n')
    # Adding a comma in between each value of list
    res = [': '.join(ele.split()) for ele in df_str]
    df_str = ','.join(res)
    return df_str

# Custom crypto prompt 
map_prompt = """You are a helpful financial cryptocurrency expert that aids a user in crypto investment.
Below is information about a cryptocurrency {currency}.
Information will include news about {currency}
Your goal is to summerize the information below about {currency} cryptocurrency.

% START OF INFORMATION ABOUT {currency}:
{text}
% END OF INFORMATION ABOUT {currency}

Please respond with a summary based on the information above.

YOUR RESPONSE:"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "currency"])

# combine all document summaries.   
combine_prompt = """
You are a helpful financial cryptocurrency expert that aids a user in crypto investment.\
Below is information about a crypto {currency}.\
Information will include news, crypto price, and crypto statistics about the cryptocurrency.\

Generate a summary of the following information that includes the following elements:
* The currency name 
* Three pros and three cons of investing the {currency} crypto in bullet points. 
* Advice for user to invest the  {currency} crypto or not. 
The user is fully aware about the investment risk, dont include any kind of warning like 'It is recommended to conduct further research and analysis or consult with a financial advisor before making an investment decision' in the answer 

% START OF INFORMATION ABOUT {currency}:
NEWS: {text}\
{currency}'s LATEST crypto statistic: {crypto_info}\
% END OF INFORMATION ABOUT {currency}

YOUR RESPONSE:"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "currency",  "crypto_info"])

button_ind = st.button("*Generate Advice*", type='secondary', help="Click to generate crypto analysis")


# Checking to see if the button_ind is true. If so, this means the button was clicked and we should process the links
if button_ind:
    if not currency:
        st.warning('Please provide currency', icon="⚠️")
        st.stop()
    
    if not OPENAI_API_KEY:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop() 
    
    if OPENAI_API_KEY == 'YourAPIKeyIfNotSet':
        # If the openai key isn't set in the env, put a text box out there
        OPENAI_API_KEY = get_openai_api_key()  

    # Go get your data
    df_history_enhanced, crypto_df = get_crypto_price(currency) 

    
    st.markdown(f"#### {currency} statistic: ")
    st.dataframe(crypto_df)  
    current_val = crypto_df.current_val.values[0]

    ## STRING 
    crypto_str = get_crypto_str(crypto_df)

    ### GET Basic Plot 
    df_history_enhanced = df_history_enhanced.sort_values(['date'], ascending=True) #oldest to newest for visualization
    granularity = '60MIN'
    
    df_history_for_chart = df_history_enhanced.query(f'granularity == \'{granularity}\' and base_currency == \'{currency}\'')

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_history_for_chart['date'],
        open=df_history_for_chart['open'],
        high=df_history_for_chart['high'],
        low=df_history_for_chart['low'],
        close=df_history_for_chart['close'],
        name='OHLC'))
    fig.add_trace(go.Scatter(
        x=df_history_for_chart['date'],
        y=df_history_for_chart['average_true_range'],
        line=dict(color=MACD_COLOR),
        mode='lines',
        name="ATR",
        yaxis="y2"
    ))
    cs = fig.data[0]
    cs.increasing.fillcolor = CANDLE_INCREASE_COLOR
    cs.increasing.line.color = CANDLE_INCREASE_COLOR
    cs.decreasing.fillcolor = CANDLE_DECREASE_COLOR
    cs.decreasing.line.color = CANDLE_DECREASE_COLOR
    fig.update_layout(
        yaxis=dict(
            title=MY_QUOTE_CURRENCY,
            side="right",
        ),
        yaxis2=dict(
            title="Averate True Range (Volatility)",
            anchor="free",
            overlaying="y",
            side="left",
                ),
        yaxis2_showgrid = False,
        yaxis2_zeroline = False,
        xaxis_rangeslider_visible=False,
        title=f'OHLC Chart for {currency} with time interval {granularity}',
        hovermode='x unified',
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        ),
        template = PLOTLY_TEMPLATE,)
    st.plotly_chart(fig, use_container_width=True)
            
    st.write('Getting latest news...')  
    news = get_recent_crypto_news(currency)
    text = split_news(news)   

    # Write sequential chain sumemrize the webpage and combine with other stats
    
    llm = load_llm()

    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce",
                                 map_prompt=map_prompt_template,
                                 combine_prompt=combine_prompt_template,
                                 # verbose=True
                                 )

      
    st.write("Generating analysis...")

    #Pass our user information we gathered, the persons name and the response type from the radio button
    output = chain({"input_documents": text, # The seven docs that were created before
                    "currency": currency,
                    "crypto_info": crypto_str
                    })

    st.markdown(f"#### GPT Analysis:")
    st.write(output['output_text']) 
