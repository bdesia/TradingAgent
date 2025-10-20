import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.signal import find_peaks

class StockFFTAnalyzer:
    def __init__(self, ticker):
        """
        Initialize the FFT analyzer with a stock ticker.
        Loads data from ../Data/<ticker>_stock_data.csv.
        
        Parameters:
        - ticker (str): Stock ticker symbol (e.g., 'SATL').
        """
        self.ticker = ticker.upper()
        self.prices = None
        self.dates = None
        self.frequencies = None
        self.fft_values = None
        self.amplitudes = None
        self.peaks = None
        self.troughs = None
        self._load_data()

    def _load_data(self):
        """
        Load stock price data from a CSV file in the ../Data folder.
        Expects data starting from row 4 with columns: Date,Close,High,Low,Open,Volume.
        Skips first three rows (incorrect header, Ticker row, empty row).
        Uses 'Close' for prices, skips invalid rows.
        """
        file_path = os.path.join("..", "Data", f"{self.ticker}_stock_data.csv")
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} not found. Use download_ticker to generate it.")
            
            # Read CSV, skip first 3 rows, assign correct column names
            data = pd.read_csv(file_path, skiprows=3, 
                             names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'], 
                             skip_blank_lines=True)
            
            # Debug: Print the first few rows and columns
            print("Loaded CSV data (first 5 rows):")
            print(data.head())
            print("Columns:", list(data.columns))
            
            # Convert Date column to datetime, coerce invalid values to NaT
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            
            # Filter out rows with invalid Close or Date values
            data = data.dropna(subset=['Date', 'Close'])
            data = data[pd.to_numeric(data['Close'], errors='coerce').notnull()]
            
            # Extract prices and dates
            self.prices = data['Close'].astype(float).values
            self.dates = data['Date'].values
            
            if len(self.prices) < 2:
                raise ValueError("At least two valid price points are required for FFT analysis.")
            if len(self.dates) != len(self.prices):
                raise ValueError("Length of dates must match length of prices.")
            
            print(f"Loaded {len(self.prices)} valid data points for {self.ticker}.")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.prices = None
            self.dates = None

    def apply_fft(self):
        """
        Apply Fast Fourier Transform to the stock price data.
        """
        if self.prices is None:
            print("No data available for FFT analysis.")
            return

        n = len(self.prices)

        # Apply FFT
        self.fft_values = np.fft.fft(self.prices)
        self.frequencies = np.fft.fftfreq(n, d=1)  # Assuming daily data (d=1 day)

        # Keep only positive frequencies (first half of the spectrum)
        self.fft_values = self.fft_values[:n//2]
        self.frequencies = self.frequencies[:n//2]

        # Compute amplitudes
        self.amplitudes = np.abs(self.fft_values) / n  # Normalize by length

    def find_peaks_and_troughs(self, distance=30):
        """
        Find peaks and troughs in the price data using scipy.signal.find_peaks.
        
        Parameters:
        - distance (int): Minimum number of days between peaks/troughs to avoid noise.
        
        Returns:
        - peaks (array): Indices of peak prices.
        - troughs (array): Indices of trough prices.
        """
        if self.prices is None:
            print("No price data available for peak/trough detection.")
            return None, None

        # Find peaks (local maxima)
        peaks, _ = find_peaks(self.prices, distance=distance)
        
        # Find troughs (local minima) by negating prices
        troughs, _ = find_peaks(-self.prices, distance=distance)
        
        self.peaks = peaks
        self.troughs = troughs
        return peaks, troughs

    def calculate_average_days(self):
        """
        Calculate the average number of days between consecutive peaks and troughs.
        
        Returns:
        - avg_days_peaks (float): Average days between consecutive peaks.
        - avg_days_troughs (float): Average days between consecutive troughs.
        """
        if self.peaks is None or self.troughs is None:
            print("Peaks and troughs not computed. Run find_peaks_and_troughs first.")
            return None, None
        
        # Calculate days between consecutive peaks
        if len(self.peaks) > 1:
            peak_dates = self.dates[self.peaks]
            peak_diffs = [(peak_dates[i+1] - peak_dates[i]) / np.timedelta64(1, 'D') for i in range(len(peak_dates)-1)]
            avg_days_peaks = np.mean(peak_diffs) if peak_diffs else np.nan
        else:
            avg_days_peaks = np.nan
            print("Not enough peaks to calculate average days.")
        
        # Calculate days between consecutive troughs
        if len(self.troughs) > 1:
            trough_dates = self.dates[self.troughs]
            trough_diffs = [(trough_dates[i+1] - trough_dates[i]) / np.timedelta64(1, 'D') for i in range(len(trough_dates)-1)]
            avg_days_troughs = np.mean(trough_diffs) if trough_diffs else np.nan
        else:
            avg_days_troughs = np.nan
            print("Not enough troughs to calculate average days.")
        
        return avg_days_peaks, avg_days_troughs

    def plot_results(self, top_n=5):
        """
        Plot the stock prices with peaks and troughs, and FFT frequency spectrum.
        
        Parameters:
        - top_n (int): Number of dominant frequencies to highlight.
        """
        if self.prices is None or self.frequencies is None:
            print("No data or FFT results to plot.")
            return

        # Plot time series with peaks and troughs
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 1, 1)
        plt.plot(self.dates, self.prices, label=f'{self.ticker} Closing Prices')
        if self.peaks is not None:
            plt.scatter(self.dates[self.peaks], self.prices[self.peaks], color='red', label='Peaks', zorder=5)
        if self.troughs is not None:
            plt.scatter(self.dates[self.troughs], self.prices[self.troughs], color='green', label='Troughs', zorder=5)
        plt.title(f'{self.ticker} Stock Prices with Peaks and Troughs')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()

        # Plot frequency spectrum
        plt.subplot(2, 1, 2)
        plt.plot(self.frequencies, self.amplitudes, label='Amplitude Spectrum')
        plt.title('FFT Frequency Spectrum')
        plt.xlabel('Frequency (cycles per day)')
        plt.ylabel('Amplitude')

        # Highlight top N frequencies
        if top_n > 0:
            top_indices = np.argsort(self.amplitudes)[-top_n:][::-1]
            top_frequencies = self.frequencies[top_indices]
            top_amplitudes = self.amplitudes[top_indices]
            plt.scatter(top_frequencies, top_amplitudes, color='red', label='Top Frequencies', zorder=5)
            
            # Annotate top frequencies with their periods (in days)
            for freq, amp in zip(top_frequencies, top_amplitudes):
                if freq > 0:  # Avoid division by zero
                    period = 1 / freq
                    plt.annotate(f'{period:.1f} days', (freq, amp), textcoords="offset points", xytext=(0,10), ha='center')

        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def get_dominant_cycles(self, top_n=5):
        """
        Return the top N dominant cycles (in days).
        
        Parameters:
        - top_n (int): Number of dominant frequencies to return.
        
        Returns:
        - List of tuples (period in days, amplitude).
        """
        if self.amplitudes is None or self.frequencies is None:
            print("No FFT results available.")
            return []

        top_indices = np.argsort(self.amplitudes)[-top_n:][::-1]
        cycles = []
        for idx in top_indices:
            freq = self.frequencies[idx]
            if freq > 0:  # Avoid division by zero
                period = 1 / freq
                amplitude = self.amplitudes[idx]
                cycles.append((period, amplitude))
        return cycles

    def analyze(self, top_n=5, peak_distance=30):
        """
        Run the full FFT analysis pipeline: apply FFT, find peaks/troughs, and plot results.
        
        Parameters:
        - top_n (int): Number of dominant frequencies to highlight in the plot.
        - peak_distance (int): Minimum days between peaks/troughs for detection.
        """
        if self.prices is None:
            print(f"No valid data loaded for {self.ticker}.")
            return
        
        print(f"Applying FFT to {self.ticker} prices...")
        self.apply_fft()
        
        print("Finding peaks and troughs...")
        self.find_peaks_and_troughs(distance=peak_distance)
        
        print("Calculating average days between peaks and troughs...")
        avg_days_peaks, avg_days_troughs = self.calculate_average_days()
        if avg_days_peaks is not None and not np.isnan(avg_days_peaks):
            print(f"Average days between peaks: {avg_days_peaks:.1f}")
        if avg_days_troughs is not None and not np.isnan(avg_days_troughs):
            print(f"Average days between troughs: {avg_days_troughs:.1f}")
        
        print("Plotting results...")
        self.plot_results(top_n)
        
        print("\nDominant cycles (period in days, amplitude):")
        for period, amplitude in self.get_dominant_cycles(top_n):
            print(f"Period: {period:.1f} days, Amplitude: {amplitude:.2f}")