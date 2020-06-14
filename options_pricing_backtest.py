import yfinance as yf
import pandas as pd
from datetime import timedelta
import numpy as np
import scipy.stats as si
import talib as ta


"""
Options backtesting model that uses
Black-Scholes options pricing model to price
the option 3% above or below (for call or put)
current price as well as the option $5 further OTM 
of the first strike to calculate the
theoretical spread credit received if the
option spread were sold and executes trade
based on oversold and overbought RSI levels
"""

class Options_Pricing(object):

    ## Initialize the stock ticker symbol, time to expiration and interest rate
    def __init__(self,ticker,exp,intRate):
        self.ticker = ticker
        self.exp = exp
        self.intRate = intRate

    ## Grabs historical data from Yahoo Finance using yfinance module
    def getPriceData(self):
        df = yf.Ticker(self.ticker)
        df = df.history(period="max")
        return df

    ## Computes RSI using ta-lib module based on Close price data
    def getRSI(self):
        df = Options_Pricing.getPriceData(self)
        RSI = ta.RSI(df["Close"], timeperiod=14)
        return RSI

    ## Computes Vol on a rolling 252 trading day window
    ## by taking the stdev of average log returns
    def getVol(self):
        df = Options_Pricing.getPriceData(self)
        close = df["Close"]
        df['Returns']= np.log(close/close.shift(1))
        sigma = df['Returns'].rolling(window=252).std() * np.sqrt(252)
        return sigma

    ## Computes two prices one for the closer to ITM call
    ## and the bottom leg of the spread
    def getCallData(self,S, date):
        bot = 0
        spread = 2
        strikes = []
        calls = []

        while spread > 0:
            K = round(S*1.03,-1)+bot
            T = self.exp
            r = self.intRate
            sigma = Options_Pricing.getVol(self)[date]
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            strikes.append(K)
            calls.append((S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)))
            spread -=1
            bot = 5
        return [round(calls[0]-calls[1],2),strikes[0],strikes[1]]

    ## Computes two prices one for the closer to ITM put
    ## and the bottom leg of the spread
    def getPutData(self,S,date):
        bot = 0
        spread = 2
        strikes = []
        puts = []

        while spread > 0:
            K = round(S-(S * .03), -1) - bot
            T = self.exp
            r = self.intRate
            sigma = Options_Pricing.getVol(self)[date]
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            strikes.append(K)
            puts.append ((K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)))
            spread -= 1
            bot = 5
        return [round(puts[0]-puts[1],2),strikes[0],strikes[1]]

class Backtest(Options_Pricing):

    ## Setup variables to test RSI, include cost of 1 contract
    ## $5 width and create lists to hold
    ## the values related to the backtest
    def runBackTest(self):
        df = Options_Pricing.getPriceData(self)
        df["RSI"] = Options_Pricing.getRSI(self)
        strikeWidth = 500
        validDates = []
        callDates, putDates = [],[]
        shortList, longList = [],[]
        shortPrice, longPrice  = [],[]
        shortStrike, longStrike = [],[]
        shortExpPrice, longExpPrice = [],[]
        shortCredit,longCredit = [],[]

        ## Checks if RSI is less than 30 or greater than 70
        ## Less than 30 == Call Spread
        ## Greater than 70 == Put Spread
        for i in df.index:
            validDates.append(i)
            if df["RSI"][i] > 70:
                shortList.append(i)
            elif df["RSI"][i] < 30:
                longList.append(i)

        ## Appends range of dates and stock price in which
        ## we are interested in backtesting as well as appends
        ## to the list the price of the stock a month out on the
        ## next Friday expiration date
        for i in shortList:
            if df["Close"][i] > 100 and df["Close"][i] < 200:
                callDates.append(i)
                shortPrice.append(df["Close"][i])
                getExp = i + timedelta(days=30)
                while getExp.weekday() != 4:
                    getExp += timedelta(days=1)
                while getExp not in validDates:
                    getExp += timedelta(days=1)
                shortExpPrice.append(df["Close"][getExp])

        ##Same premise as above but for put spreads
        for i in longList:
            if df["Close"][i] > 100 and df["Close"][i] < 200:
                putDates.append(i)
                longPrice.append(df["Close"][i])
                getExp = i + timedelta(days=30)
                while getExp.weekday() != 4:
                    getExp += timedelta(days=1)
                while getExp not in validDates:
                    getExp += timedelta(days=1)
                longExpPrice.append(df["Close"][getExp])

        ## Computes the spread price for a credit call spread
        ## given the parameters being met
        ## For testing purposes this is the first 15 trades
        for i in shortPrice[0:15]:
            w = shortPrice.index(i)
            Date = validDates.index(callDates[w])
            callData = Options_Pricing.getCallData(self,i,Date)
            shortCredit.append(callData[0]*100)
            credit = callData[0]*100
            shortStrike.append(callData[2])
            print(f'Sold credit call spread at {callData[1]},{callData[2]} for ${round(credit,2)} credit')

        ## Computes the spread price for a credit put spread
        ## given the parameters being met
        ## For testing purposes this is the first 15 trades
        for i in longPrice[0:15]:
            y = longPrice.index(i)
            Date2 = validDates.index(putDates[y])
            putData = Options_Pricing.getPutData(self,i,Date2)
            longCredit.append(putData[0]*100)
            credit = putData[0]*100
            longStrike.append(putData[2])
            print(f'Sold credit put spread at {putData[1]},{putData[2]} for ${round(credit,2)} credit')

        print("Calculating return...")

        ## Reduces each element in the longCredit received list
        ## by the strike width if a loss as to simulate a max loss
        for i in shortCredit:
            k = shortCredit.index(i)
            if shortStrike[k] > shortExpPrice[k]:
                shortCredit[k] = shortCredit[k]
                print(f"Profit of {shortCredit[k]}")
            else:
                shortCredit[k] -= strikeWidth
                print(f"Loss of {shortCredit[k]}")

        ## Reduces each element in the shortCredit received list
        ## by the strike width if a loss as to simulate a max loss
        for i in longCredit:
            k = longCredit.index(i)
            if longStrike[k] < longExpPrice[k]:
                longCredit[k] = longCredit[k]
                print(f"profit of {longCredit[k]}")
            else:
                longCredit[k] -= strikeWidth
                print(f"loss of {longCredit[k]}")

        ## Sum each elements of the lists then combine to
        ## arrive at pnl
        pnl = sum(shortCredit)+sum(longCredit)
        print(f"Calls from {callDates[0]} to {callDates[15]}...")
        print(f"Puts from {putDates[0]} to {putDates[15]}...")

        return f"Profit/loss is ${pnl} from collateral of ${(len(shortPrice[0:15])+len(longPrice[0:15]))*strikeWidth}"

exp = 1/12
intRate = .01

AAPL = Backtest("AAPL",exp,intRate)
print(AAPL.runBackTest())







