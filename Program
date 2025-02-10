import numpy as np
import pandas as pd

# This program aims to recreate Jegadeesh and Titman momentum experience, consisting in buying winner and selling loser portfolios
# The data set covers the same historical from 1965 to 1989, just like the original work
# The user can define the analysis period (default 12 months = 2 semesters) and the holding period (default 6 months = 1 semester)

ANALYSIS_PERIOD = 2  # Number of semesters used to classify portfolios (12 months = 2 semesters)
HOLDING_PERIOD = 1   # Number of semesters the portfolios are held (6 months = 1 semester)

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    
    # Drop missing values
    df.dropna(inplace=True)

    # Rename columns
    df = df.rename(columns={'date': 'Date', 'PRIMEXCH': 'Prim_Exch', 
                            'RET': 'Returns', 'PERMNO': 'Permno',  
                            'semester': 'Semester'})
    
    # Keep only NYSE and AMEX stocks
    df = df[df['Prim_Exch'].isin(['N', 'A'])]

    # Remove non-numeric values from Returns column
    df = df[df['Returns'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]

    # Convert Returns and Semester to appropriate types
    df['Returns'] = df['Returns'].astype(float)
    df['Semester'] = df['Semester'].astype(int)

    # Sort by security and date
    df = df.sort_values(by=['Permno', 'Date'])

    # Getting smestrial returns
    df = df.groupby(['Permno', 'Semester'])['Returns'].sum().reset_index()
   
    return df

def compute_cumulative_returns(df):

    df_cumulative = df.copy()   

    # Apply a rolling sum over 2 semesters to get the yearly returns and rank the portfolios
    df_cumulative['Rolling_Cumulative_Returns'] = df_cumulative.groupby('Permno')['Returns'].rolling(
        ANALYSIS_PERIOD).sum().reset_index(level=0, drop=True)

    # Deciles calculation, based on returns over the year
    df_cumulative['Decile'] = df_cumulative.groupby('Semester')['Rolling_Cumulative_Returns'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

    # Merge with the original dataset
    df = df.merge(df_cumulative[['Permno', 'Semester', 'Rolling_Cumulative_Returns', 'Decile']], 
                  on=['Permno', 'Semester'], how='left')
    
    return df

def identify_winners_and_losers(df):
  
    # Get winners (top 10%) and losers (bottom 10%)
    winners = df[df['Decile'] == 9].copy()
    losers = df[df['Decile'] == 0].copy()
    
    # Define the next semester for performance tracking
    winners['Next_Semester'] = winners['Semester'] + HOLDING_PERIOD
    losers['Next_Semester'] = losers['Semester'] + HOLDING_PERIOD

    return winners, losers

def compute_future_returns(df, winners, losers):

    df_next_period_perf = df[['Permno', 'Semester', 'Returns']].copy()
    df_next_period_perf = df_next_period_perf.rename(columns={'Returns': 'Next_Semester_Returns'})

    # Merge future performance
    winners = winners.merge(df_next_period_perf, left_on=['Permno', 'Next_Semester'], right_on=['Permno', 'Semester'], how='left')
    losers = losers.merge(df_next_period_perf, left_on=['Permno', 'Next_Semester'], right_on=['Permno', 'Semester'], how='left')
   
    # Clean columns
    winners.drop(columns=['Semester_y'], inplace=True)
    losers.drop(columns=['Semester_y'], inplace=True)
    winners.rename(columns={'Semester_x': 'Semester'}, inplace=True)
    losers.rename(columns={'Semester_x': 'Semester'}, inplace=True)

    return winners, losers

def run_program(file_path):
    df = load_data(file_path)
    df = compute_cumulative_returns(df)
    winners, losers = identify_winners_and_losers(df)
    winners, losers = compute_future_returns(df, winners, losers)

    # Compute mean returns of winners and losers
    mean_winner_returns = winners['Next_Semester_Returns'].mean()
    mean_loser_returns = losers['Next_Semester_Returns'].mean()

    # Calculate momentum premium
    momentum_premium = mean_winner_returns - mean_loser_returns

    print(f"Average return of Winners (next {HOLDING_PERIOD * 6} months): {mean_winner_returns:.2%}")
    print(f"Average return of Losers (next {HOLDING_PERIOD * 6} months): {mean_loser_returns:.2%}")
    print(f"Momentum Premium (Winners - Losers): {momentum_premium:.2%}")

run_program('CRSP-data.csv')
