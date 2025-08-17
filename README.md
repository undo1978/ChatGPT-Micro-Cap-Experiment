# ChatGPT Micro-Cap Experiment
Welcome to the repo behind my 6-month live trading experiment where ChatGPT manages a real-money micro-cap portfolio.

### Frontend Help Wanted!!!

I’m currently looking for **frontend contributors** to help improve the project’s UI.  

Areas where help is needed:  
- Graph rendering (fixing data points not loading properly)  
- Email login page (debugging and styling issues)  
- General UI/UX improvements  

Check out the [Contributing Guide](https://github.com/LuckyOne7777/ChatGPT-Portfolio-Overhaul/blob/main/CONTRIBUTING.md) to learn how to get involved.


# The Concept
Every day, I kept seeing the same ad about having some A.I. pick undervalued stocks. It was obvious it was trying to get me to subscribe to some garbage, so I just rolled my eyes.  
Then I started wondering, "How well would that actually work?"

So, starting with just $100, I wanted to answer a simple but powerful question:

**Can powerful large language models like ChatGPT actually generate alpha (or at least make smart trading decisions) using real-time data?**

## Each trading day:

- I provide it trading data on the stocks in its portfolio.  
- Strict stop-loss rules apply.  
- Every week I allow it to use deep research to reevaluate its account.  
- I track and publish performance data weekly on my blog: [Here](https://nathanbsmith729.substack.com)

## Research & Documentation

- [Research Index](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Deep%20Research%20Index.md)  
- [Disclaimer](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Disclaimer.md)  
- [Q&A](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Q%26A.md)  
- [Prompts](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Prompts.md)  
- [Starting Your Own](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Start%20Your%20Own/README.md)  
- [Research Summaries (MD)](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/tree/main/Weekly%20Deep%20Research%20(MD))  
- [Full Deep Research Reports (PDF)](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/tree/main/Weekly%20Deep%20Research%20(PDF))  

# Performance Example (6/30 – 7/25)

---
<!-- To update dates (%286-30%20-%208-15%29%20Results), change the "8-15" in the middle. -->
![Week 7 Performance](%286-30%20-%208-15%29%20Results.png)

---
- Currently outperforming the S&P 500.

# Features of This Repo
- Live trading scripts — used to evaluate prices and update holdings daily  
- LLM-powered decision engine — ChatGPT picks the trades  
- Performance tracking — CSVs with daily PnL, total equity, and trade history  
- Visualization tools — Matplotlib graphs comparing ChatGPT vs. Index  
- Logs & trade data — auto-saved logs for transparency  

# Why This Matters
AI is being hyped across every industry, but can it really manage money without guidance?

This project is an attempt to find out — with transparency, data, and a real budget.

# Tech Stack
- Basic Python  
- Pandas + yFinance for data & logic  
- Matplotlib for visualizations  
- ChatGPT-5 for decision-making  

# Installation
To run the scripts locally, install the Python dependencies:

```bash
pip install -r requirements.txt
```

# Follow Along
The experiment runs from June 2025 to December 2025.  
Every trading day I will update the portfolio CSV file.  
If you feel inspired to do something similar, feel free to use this as a blueprint.

Updates are posted weekly on my blog — more coming soon!

One final shameless plug: (https://substack.com/@nathanbsmith?utm_source=edit-profile-page)

Find a mistake in the logs or have advice?  
Please reach out here: **nathanbsmith.business@gmail.com**
