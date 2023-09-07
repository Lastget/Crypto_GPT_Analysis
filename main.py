from googlesearch import search
import os
import time
from bs4 import BeautifulSoup
import re
import requests
import warnings

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

# check Coinbase key 

# Start streamlit 
st.set_page_config(page_title="Crypto Analysis", page_icon="🤑")

# Start top information 
st.header("GPT Crypto Analysis 🤑")

st.markdown("You want to save time analyze a cryptocurrency? This tool is meant to help you decide whether to buy a crypto or not.\
            \n\nThis tool is powered by [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/#), [markdownify](https://pypi.org/project/markdownify/), [LangChain](https://langchain.com/), and [OpenAI](https://openai.com)" \
            )

# Collect information about the person you want to research
currency = st.text_input(label="currency",  placeholder="Ex: ETH", key="currency")

# Fetch stock data from Coinbase 
def get_stock_price(currency,history=5):
    ''' 
      input: 
        - currency 
        - history: how many days of stock data to look at.  
      output:
        - close price for the stock -> (str)
    '''
    # Avoid rate limit error
    # time.sleep(4) 
    stock = yf.currency(currency)
    df = stock.history(period="1y")
    df=df[["Close"]]
    # Get the date 
    df.index=[str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date",inplace=True)
    df.reset_index(inplace=True)
    df = df[-history:]
    return df

def get_stock_str(df):
    df_str = df.to_string(index=False, header=False, col_space=None).split('\n')
    # Adding a comma in between each value of list
    res = [': '.join(ele.split()) for ele in df_str]
    df_str = ','.join(res)
    return df_str

# Script to scrap top5 google news for given company name
def get_recent_stock_news(currency):
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
# Fetch financial statements from Yahoo Finance
def get_financial_statements(currency):
    # time.sleep(4) #To avoid rate limit error   
    company = yf.currency(currency)
    
    # get revenue, income, eps from latest quarterly income statement 
    income_stmt_table = company.quarterly_income_stmt.loc[['Total Revenue','Net Income','Diluted EPS']]
    income_stmt_table = income_stmt_table.iloc[:,0]
    total_rev = income_stmt_table['Total Revenue']
    net_income = income_stmt_table['Net Income'] 
    diluted_eps = income_stmt_table['Diluted EPS'] 

    # Get ROE and profit margin from latest quarterly balance sheet
    balance_stmt_table = company.quarterly_balance_sheet.loc['Stockholders Equity']
    st_equity = balance_stmt_table[0] 
    roe = round(net_income/st_equity,3)
    profit_margin = round(net_income/total_rev *100,3)

    # return a short summery 
    fin_stat = f"Total revenue:{total_rev} \n Net income:{net_income}\n Diluted EPS:{diluted_eps}\n ROE:{roe}\n Profit margin:{profit_margin}%"
    return fin_stat, profit_margin, total_rev, net_income, diluted_eps, roe

# Function to change long text about a topic into document 
def split_news(info):
    # make splitter 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=50)

    # split info into docs 
    spt_news  = text_splitter.create_documents([info])

    return spt_news 

# Custom stock prompt 
map_prompt = """You are a helpful financial expert that aids a user in stock investment.
Below is information about a stock {currency}.
Information will include news about {currency}
Your goal is to summerize the information below about {currency} stock.

% START OF INFORMATION ABOUT {currency}:
{text}
% END OF INFORMATION ABOUT {currency}

Please respond with a summary based on the information above.

YOUR RESPONSE:"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "currency"])

# combine all document summaries.   
combine_prompt = """
You are a helpful financial expert that aids a user in stock investment.\
Below is information about a stock {currency}.\
Information will include news, stock price, and financial statistics about the stock.\

Generate a summary of the following information that includes the following elements:
* The company name and currency of the stock. 
* Three pros and three cons of investing the stock in bullet points. 
* Advice for user to invest the stock or not. 
The user is fully aware about the investment risk, dont include any kind of warning like 'It is recommended to conduct further research and analysis or consult with a financial advisor before making an investment decision' in the answer 

% START OF INFORMATION ABOUT {currency}:
NEWS: {text}\
{currency}'s LATEST STOCK PRICE: {stock_price}\
{currency}'s FINANCIAL STATISTICS: {fin_stat}
% END OF INFORMATION ABOUT {currency}

YOUR RESPONSE:"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "currency",  "stock_price", "fin_stat"])

button_ind = st.button("*Generate Advice*", type='secondary', help="Click to generate stock analysis")


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
    stock_df = get_stock_price(currency) 
    stock_price = get_stock_str(stock_df)
    st.markdown(f"#### Stock price: ")
    st.line_chart(stock_df, x='Date', y='Close')

    stats, profit_margin, total_rev, net_income, diluted_eps, roe =  get_financial_statements(currency)
    st.markdown(f"#### Stock Stats: ")
    st.markdown(f"##### Profit margin: ")
    st.write(profit_margin) 
    st.markdown(f"##### Diluted EPS: ")
    st.write(diluted_eps) 
    st.markdown(f"##### Net income: ")
    st.write(net_income) 
    st.markdown(f"##### Total revenue: ")
    st.write(total_rev) 
    st.markdown(f"##### ROE: ")
    st.write(roe) 

    
    st.write('Getting latest news...')  
    news = get_recent_stock_news(currency)
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
                    "fin_stat": stats,
                    "stock_price": stock_price,
                    })

    st.markdown(f"#### GPT Analysis:")
    st.write(output['output_text']) 
