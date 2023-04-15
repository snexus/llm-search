Editor's note: Seeking Alpha is proud to welcome Envision Research as a new contributor. It's easy to become a Seeking Alpha contributor and earn money for your best investment ideas. Active contributors also get free access to SA Premium. [Click here to find out more »](https://seekingalpha.com/page/become-a-seeking-alpha-contributor)

![The New York Times 2014 DealBook Conference](https://static.seekingalpha.com/cdn/s3/uploads/getty_images/460307658/medium_image_460307658.jpg)

Photo by Thos Robinson/Getty Images Entertainment via Getty Images

## Dalio's All Weather Portfolio

Ray Dalio's All Weather Portfolio (AWP) is a diversified portfolio consisting of four asset classes (stocks, bonds, commodities, and gold). The idea is based on risk parity, and it has been shown to be very effective in limiting volatility and drawdowns. The name itself suggests the essence (or the goal) - it is a portfolio built to weather any storm. More specifically, the original AWP consists of:

-   30% US stocks
-   40% long-term treasuries
-   15% intermediate-term treasuries
-   7.5% commodities, diversified
-   7.5% gold

## Practical Issues

In practice, there are however several issues in the implementation of the idea. First, it is difficult, at least for individual investors, to invest in commodities with reasonable cost/fees. Second, it might be too conservative for many investors (with only 30% stocks). I wouldn't adopt it myself now. I may consider it when I get really close to my retirement age. Third, the fixed allocation does not make intuitive sense. As a recent example, during the 2020 March COVID-19 crash, treasury yields dropped to a level (0.6% for 10 year maturity) that Dalio himself considered silly to own government bonds. Yet if you stick to the original AWP, you would still be holding 55% of government bonds in your portfolio.

## AWP With Dynamic Leveraging

Being a big fan of the AWP idea (I truly believe the underlying idea is sound) and recognizing the above issues, I have been researching ways to overcome the above issues and revise it to a version that is more applicable to me. At this point, I finally figured out a revision that I am comfortable with for the long run. I will elaborate on my own financial situation and investment goals at the end so you can evaluate whether this approach applies to you. I have been implementing it for about 2 years (even, or should I say especially, during the 2020 March turmoil). For the commodity issue, I will just not use it (a really simple solution). The major revision involves applying leverage and dynamically adjust it based on volatility. The key revisions are:

1\. Use 30% as the floor for stock allocation, and add to it based on the yield spread between the S&P 500 (earning divided by price) and 10 year treasury yield. So stock allocation is at least 30%, and increases as stocks become less expensive compared to bonds. Gold is still at 7.5%. And bonds will be whatever the remaining percentage is.

2\. Apply a leverage to the stock holdings. In my backtest, I used two stock ETFs: SPDR S&P 500 Trust ETF ([SPY](https://seekingalpha.com/symbol/SPY?source=content_type%3Areact%7Csection%3Amain_content%7Cbutton%3Abody_link "SPDR S&P 500 Trust ETF")) and ProShares Ultra S&P500 (NYSEARCA:[SSO](https://seekingalpha.com/symbol/SSO?source=content_type%3Areact%7Csection%3Amain_content%7Cbutton%3Abody_link "ProShares Trust - ProShares Ultra S&P500")) (the 2x leveraged version of SPY) due to their longer history so I can backtest more scenarios. The relative weight of SSO was dynamically adjusted according to 6/VIX, with VIX being the CBOE Volatility Index. So for example, when VIX = 24, the stock allocation would consist of 25% SSO and 75% SPY. The constant obviously can be adjusted to accommodate each person's different risk level. But the idea is to leverage less (i.e., hold less SSO) when VIX increase, and vice versa. This idea of using more leverage when volatility is low has been thoroughly discussed and tested elsewhere ([for example](https://tinyurl.com/f9vr88ss), see "Alpha Generation and Risk Smoothing Using Managed Volatility" by Tony Cooper) and won't be further elaborated on here.

3\. Then adjust the portfolio based on items 1 and 2 periodically.

## Backtesting Results

The above idea has been systematically backtested over different periods of times (bull market, range-bound market, and bear market), with different levels of adjustment to the stock allocation and different levels of leverage. All results show that it can outperform the overall market AND also outperform the unleveraged AWP systematically when implemented over a minimum of a few years. The backtest used SPY and SSO to represent the unleveraged and leveraged stock fund. The reason for using them is that they have the longest history, and as such I could perform backtesting over a longer period of times and under more different scenarios. I do not actually hold SPY and SSO in my portfolio. (My actual portfolio holdings are listed at the end.)

The backtest result during a 15-year period, from 2006 to 2020 (SSO was created in June 2006), is shown below in Fig. 1. As you can see, the backtest compared the leveraged AWP method against the original AWP and the S&P 500 (represented by the Vanguard 500 index fund). The portfolio was adjusted monthly, return included dividend, and leverage factor was 6/VIX. The leveraged AWP outperformed both the AWP and the S&P 500 most of the time, and really took off and began to dominate them shortly after the 2008 great crash was over.

Overall, the leverage AWP outperformed both the AWP and the S&P 500 with a significant margin during this period (14.45% CAGR compared to 8.x%). Table 1 below provides a more detailed look. As seen, the leveraged AWP also provided lower risks, as measured by standard deviation, worst year loss, Sharpe Ratio, Sortino Ratio, et al. It only suffered a maximum drawdown that was slightly worse than the AWP ( -30.6% vs -27.02%), but still significantly better than the S&P 500 (-50.97%).

[![](https://static.seekingalpha.com/uploads/2021/5/17/48844541-16212820176170907.png)](https://static.seekingalpha.com/uploads/2021/5/17/48844541-16212820176170907_origin.png)

Source: Author, with simulator from Portfolio Visualizer, Silicon Cloud Technologies LLC

[![](https://static.seekingalpha.com/uploads/2021/5/17/48844541-1621282018399932.png)](https://static.seekingalpha.com/uploads/2021/5/17/48844541-1621282018399932_origin.png)

Source: Author, with simulator from Portfolio Visualizer, Silicon Cloud Technologies LLC

To examine the above results more closely, Fig. 2 and Table 2 below show the comparison in a shorter period of time, during 2007 to 2011, for several purposes. First, the domination of the leverage AWP during most of the period in Fig. 1 makes it difficult to see some of the details. Second, many investors probably do not worry too much during a long bull market (like the one since 2009). As long as the portfolio is making money, many investors probably do not concern too much if it makes 8% a year or 14% a year (which is actually a wise philosophy). It is in turbulent times that many investors panic and deviate from their strategy and lose money.

The results below show the comparison as a stress test to the method. As seen, the leveraged AWP outperformed both the AWP and S&P 500 most of the time during this extremely turbulent time in terms of return. Again, taking a closer look as shown in table 2 below, the leveraged AWP also provided lower risks as measured by most of the risk metrics such as standard deviation and worst year loss. The only exception again was the maximum drawdown, again slightly worse than the AWP but still significantly better than the S&P 500. This result illustrates the effectiveness of the leveraged AWP and dynamics allocations to weather storms during a period of 5 years when the overall market essentially went nowhere, with a 50% crash in the middle. Of course, the result also illustrate that the AWP is very effective by itself during turbulent times like these.

[![](https://static.seekingalpha.com/uploads/2021/5/17/48844541-16212820184339619.png)](https://static.seekingalpha.com/uploads/2021/5/17/48844541-16212820184339619_origin.png)

Source: Author, with simulator from Portfolio Visualizer, Silicon Cloud Technologies LLC

[![](https://static.seekingalpha.com/uploads/2021/5/17/48844541-16212820183546717.png)](https://static.seekingalpha.com/uploads/2021/5/17/48844541-16212820183546717_origin.png)

Source: Author, with simulator from Portfolio Visualizer, Silicon Cloud Technologies LLC

## How I Use the Strategy

First, some background about my own financial situations and goals. I am in my mid-forties with a super stable job. My wife also earns a stable professional income. So I can afford some risk taking. I have also been researching and practicing quantitative portfolio management since 2006 (shortly after I learned about it in graduate school). I have gone through two crashes (2008 and 2020), and have had the practice and experiences of actually taking risks. Knowing what to do and actually doing it are different, especially with quite a bit at stake during trying times.

With the above background, here is what I actually hold in my portfolio. As mentioned above, I do not actually hold SPY and SSO in my portfolio. I only used them in the backtesting because they have the longest history. What I do hold are Vanguard Total Stock Market Index Fund ETF Shares ([VTI](https://seekingalpha.com/symbol/VTI?source=content_type%3Areact%7Csection%3Amain_content%7Cbutton%3Abody_link "Vanguard Total Stock Market ETF")) and ProShares UltraPro QQQ ([TQQQ](https://seekingalpha.com/symbol/TQQQ?source=content_type%3Areact%7Csection%3Amain_content%7Cbutton%3Abody_link "ProShares UltraPro QQQ ETF")). I prefer VTI over SPY mostly for its broader market representation and slightly lower fee. I prefer TQQQ over SSO for two main reasons. First, it has a better liquidity. Second, TQQQ provides 3x leverage. As a result, I could adjust my leverage factor from 6/VIX to 4/VIX and hold lower percentage of TQQQ to provide the same amount of leverage, which helps to reduce the fee drag, dividend drag, and also to reduce my absolute dollar amount exposed to leveraging. Lastly, I also cap my leverage to the entire portfolio to 25%.

So, as a hypothetical example, if VIX ever drops to 8 I would still leverage to 25%, not 50%. My plan/goal is to keep managing our accounts this way for the next 10 years, then gradually transition into a less aggressive strategy when we get closer to retirement. Note the degree of aggressiveness is entirely relative. My current leveraged AWP approach (especially with the 25% cap) is considered conservative by some of my colleagues and friends. And my less aggressive plan 10 years later might be considered too aggressive to many of you (the plan involves applying the leveraged strategy but with a few years of cushion fund and a lower leverage factor).

## And What About You?

Whether the above approach applies to you or not of course depends on your own financial situations, goals, and risk tolerance. Here are a few things that I've learned and you could consider.

First, like many of you, I started my investment journey with a belief in the "common sense" that leveraged ETFs are bad for long-term holding, which really is only a myth after careful thinking and examination. Others (see "Alpha Generation and Risk Smoothing using Managed Volatility," linked to above) have done an excellent job examining (and busting) this myth. The upshot is that there is nothing wrong of holding a leveraged ETF for the long term as long as a) you can tolerate the risk of total loss (e.g., if you hold TQQQ, a 33.3% daily crash would lead to total loss since it is 3x leveraged), b) the boost in return can compensate the fees charged by leveraged ETFs (the fee drag), and c) the boost in return can compensate the loss of dividend due to holding leveraged ETFs (the dividend drag). The fee drag and dividend drag are not too large to start with in my current setup (but not negligible either). With the 4/VIX leverage factor, the weight of TQQQ in the overall portfolio is on average 15%. With a ~0.9% fee (compared to ~ 0.03% fee for unleveraged ETF like VTI), the fee drag is about ~0.13% on the overall portfolio. And the dividend drag is about 0.15%, assuming an average yield of 1.8% from VTI and 0% from TQQQ.

Second, besides the fee drag and dividend drag, leveraged ETFs also suffer a volatility drag. As an example, if the unleveraged underlying asset price (e.g., SPY) goes up 1% on day 1 and down 1% on day 2, the overall price would _not_ return to the original price on day 1, but would go _down_ by 0.01%. It is purely math - because (1+1%)\*(1-1%) is less than 1 by 0.01%. And that 0.01% is the volatility drag in this example. _All_ ETFs suffer from volatility drag, and it is just that leveraged ETFs suffer more. Following this example where SPY fluctuates 1% and -1% for two days, the 2x leveraged SSO would go up 2% on day 1 and go down 2% on day 2, and the volatility drag would be 0.04% (because (1+2%)\*(1-2%) is less than 1 by 0.04%). Note that 0.04% is not only larger than that of the underlying, but also larger than 2x of it.

As a result, when the overall market is range bound or suffers a one-way decline for an extended period of time, the leveraged asset would not only underperform the unleveraged asset, but underperform it more than the nominal leverage implies (both due to the fee drag, dividend drag, and the volatility drag). That is part of the reason for the strategy's underperformance during 2007 to 2008 compared to the original AWP. As a reward, if the opposite happens (i.e., when the overall market enjoys a one-way rally) leveraged ETFs would return more - not only more than the unleveraged underlying, but also more than their nominal leverage implies (i.e., a volatility boost). For example, if SPY goes up 1% on day 1 and another 1% on day 2, its total return would be 2.01% (because (1+1%)\*(1+1%) is larger than 1 by 2.01%). But SSO would return 4.04% (because (1+2%)\*(1+2%) is larger than 1 by 4.04%), which is not only more than 2.01%, but also more than 2x2.01% (after all, math is fair).

To summarize, the major risks associated with the above leveraged AWP approach are the possibility of total loss of the leveraged asset when 1-day crash exceeds than 33.3% if you use 3x leveraged ETFs, the fee drag due to higher fee charged by leveraged ETFs, the dividend drag due to the lack of dividend from leveraged ETFs, and the volatility drag if the market is range bound or suffers directional decline over an extended period of time.

If you can tolerate the above risks and want to consider the leveraged AWP approach, note that you have several knobs to further fine tune the approach to suit your needs. The most effective knob is the constant used in the leverage factor. As mentioned, I use 4/VIX. You could use a smaller or larger constant to tune up or down the leverage. A more sophisticated model would be to adjust the leverage factor by A/VIX^n, so you have two parameters to adjust: a constant A and an exponent n. You can adjust both the magnitude and also the sensitivity of leverage. A larger n, say n=2, would lead to adjustment that are more sensitive to VIX (e.g., in case you are interested in more frequent trades/rebalances), and a smaller n, say n=1/2, would lead to the opposite.

Lastly, in case you're wondering why I do not simply hold a risk parity ETF like RPAR as a short cut to accomplish what I attempt to do, my reasons are threefold. First, RPAR does not provide leverage (at least does not provide it with the flexibility I want). Second, the fee of RPAR (or other similar multi-asset funds) is higher than my DIY portfolio. And lastly, my DIY portfolio also provides better liquidity (each component in the DIY portfolio trades with far better liquidity than RPAR).

This article was written by

[![Envision Research profile picture](https://static1.seekingalpha.com/images/users_profile/048/844/541/medium_pic.png)](https://seekingalpha.com/author/envision-research?source=content_type%3Areact%7Cfirst_level_url%3Aarticle%7Csection%3Amain_content%7Csection_asset%3Aauthor_follow_bottom%7Cbutton%3Aavatar)

\*\* Disclosure: I am associated with Sensor Unlimited.\*\* Master of Science, 2004, Stanford University, Stanf...

Value, Deep Value, Growth At A Reasonable Price, Long Only

Contributor Since 2017

\*\* Disclosure: I am associated with Sensor Unlimited.

\*\* Master of Science, 2004, Stanford University, Stanford, CA   

Department of Management Science and Engineering, with concentration in quantitative investment 

\*\* 15 years of investment management experiences. 

Since 2006, have been actively analyzing stocks and the overall market, managing various portfolios and accounts and providing investment counseling to many relatives and friends.

\*\* Writing interest - Long term portfolio management, quantitative portfolio management, selection of value stocks, dividend stocks, personal finances, investment psychology

\*\* And most importantly, write to share and exchange lessons I've learned since I started investing in 2006, and to learn new lessons from this wonderful community

**Disclosure:** I am/we are long VTI TQQQ. I wrote this article myself, and it expresses my own opinions. I am not receiving compensation for it (other than from Seeking Alpha). I have no business relationship with any company whose stock is mentioned in this article.