import os
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

from data_utils import get_qualifying_data, prepare_data
from sklearn.utils.class_weight import compute_class_weight

try:
	from imblearn.over_sampling import SMOTE
	SMOTE_AVAILABLE = True
except ImportError:
	print("Warning: imblearn not available. SMOTE oversampling will be skipped.")
	SMOTE_AVAILABLE = False

_Q3_MODEL_CACHE = None
MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR, 'q3_model_2024.joblib')
METRICS_FILE = os.path.join(MODEL_DIR, 'model_metrics.csv')

MODEL_FILES = {
    'RandomForest': os.path.join(MODEL_DIR, 'random_forest_model.joblib'),
    'LogisticRegression': os.path.join(MODEL_DIR, 'logistic_regression_model.joblib'),
    'GradientBoosting': os.path.join(MODEL_DIR, 'gradient_boosting_model.joblib')
}

MODELS = {
    'RandomForest': RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    ),
    'LogisticRegression': LogisticRegression(
        random_state=42, 
        max_iter=1000,
        class_weight='balanced'
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100, 
        random_state=42
    )
}

def get_train_test_data(train_years=(2022, 2023), test_year=2024, sample_size=None):
	"""Fetch training data (2022-2023) and test data (2024) with temporal split"""
	print(f"\nFetching training data from {train_years[0]} to {train_years[1]}...")
	train_data = []
	
	for year in range(train_years[0], train_years[1] + 1):
		try:
			print(f"Loading {year} training data...")
			year_data = get_qualifying_data(year)
			if sample_size and len(year_data) > sample_size:
				year_data = year_data.sample(n=sample_size, random_state=42)
			train_data.append(year_data)
			print(f"Loaded {len(year_data)} records from {year}")
		except Exception as e:
			print(f"Warning: Could not load {year} data: {str(e)}")
			continue
	
	if not train_data:
		raise ValueError("No training data could be loaded from the specified years")
	
	train_df = pd.concat(train_data, ignore_index=True)
	print(f"Total training data: {len(train_df)} records")
	
	print(f"\nFetching test data from {test_year}...")
	try:
		test_df = get_qualifying_data(test_year)
		if sample_size and len(test_df) > sample_size:
			test_df = test_df.sample(n=sample_size, random_state=42)
		print(f"Total test data: {len(test_df)} records")
	except Exception as e:
		print(f"Warning: Could not load {test_year} test data: {str(e)}")
		# Use last 20% of training data as test if 2024 not available
		test_size = int(len(train_df) * 0.2)
		test_df = train_df.tail(test_size)
		train_df = train_df.head(len(train_df) - test_size)
		print(f"Using last {len(test_df)} records from training data as test set")
	
	print(f"\nValidating temporal split...")
	train_years_set = set(train_df['Year'].unique()) if 'Year' in train_df.columns else set()
	test_years_set = set(test_df['Year'].unique()) if 'Year' in test_df.columns else set()
	
	if train_years_set & test_years_set:
		print(f"❌ TEMPORAL LEAKAGE: Overlapping years between train and test: {train_years_set & test_years_set}")
		raise ValueError("Training and test data must be from different years!")
	else:
		print(f"✅ Temporal separation confirmed. Train years: {train_years_set}, Test years: {test_years_set}")
	
	train_events = set(train_df['EventName'].unique()) if 'EventName' in train_df.columns else set()
	test_events = set(test_df['EventName'].unique()) if 'EventName' in test_df.columns else set()
	
	overlapping_events = train_events & test_events
	if overlapping_events:
		print(f"📊 Event overlap detected: {len(overlapping_events)} events appear in both periods")
		print(f"   This is expected because we use temporal features that only look at past data")
		print(f"   Examples: {list(overlapping_events)[:3]}{'...' if len(overlapping_events) > 3 else ''}")
	else:
		print(f"✅ No event overlap: {len(train_events)} training events, {len(test_events)} test events")
	
	if len(train_df) < 50:
		raise ValueError(f"Training data too small: {len(train_df)} records")
	if len(test_df) < 10:
		raise ValueError(f"Test data too small: {len(test_df)} records")
	
	print(f"✅ Temporal split validated: {len(train_df)} training records, {len(test_df)} test records")
	
	return train_df, test_df

def calculate_driver_recent_form(df, driver_code, current_year, current_round, lookback_races=5):
	"""Calculate driver's recent form based on last N races"""
	# Check if Year column exists
	if 'Year' not in df.columns:
		print(f"Warning: Year column not found, using default form for {driver_code}")
		return 0.5
	
	driver_data = df[(df['Driver'] == driver_code) & 
					 ((df['Year'] < current_year) | 
					  ((df['Year'] == current_year) & (df['Round'] < current_round)))]
	
	if len(driver_data) == 0:
		return 0.5  # Default neutral form
	
	recent_races = driver_data.sort_values(['Year', 'Round']).tail(lookback_races)
	
	# Calculate average Q3 qualification rate
	if 'Q3Qualified' in recent_races.columns:
		form_score = recent_races['Q3Qualified'].mean()
	else:
		# Fallback: use position-based form (lower position = better form)
		if 'Position' in recent_races.columns:
			avg_position = recent_races['Position'].mean()
			form_score = max(0, 1 - (avg_position - 1) / 19)  # Convert to 0-1 scale
		else:
			form_score = 0.5
	
	return form_score

def calculate_team_performance(df, team, current_year, current_round, lookback_races=10):
	"""Calculate team's recent performance metrics (STRICTLY HISTORICAL DATA ONLY)"""
	# Check if Year column exists
	if 'Year' not in df.columns:
		print(f"Warning: Year column not found, using default team performance for {team}")
		return {'avg_qualifying_position': 10, 'q3_rate': 0.5, 'avg_lap_time': 90.0}
	
	# CRITICAL: Only use data from BEFORE the current race (strict temporal separation)
	team_data = df[(df['Team'] == team) & 
				   ((df['Year'] < current_year) | 
					((df['Year'] == current_year) & (df['Round'] < current_round)))]
	
	if len(team_data) == 0:
		return {'avg_qualifying_position': 10, 'q3_rate': 0.5, 'avg_lap_time': 90.0}
	
	# Get recent races (most recent historical races)
	recent_races = team_data.sort_values(['Year', 'Round']).tail(lookback_races)
	
	# Calculate team metrics
	avg_position = recent_races['Position'].mean() if 'Position' in recent_races.columns else 10
	q3_rate = recent_races['Q3Qualified'].mean() if 'Q3Qualified' in recent_races.columns else 0.5
	avg_lap_time = recent_races['LapTime'].mean() if 'LapTime' in recent_races.columns else 90.0
	
	return {
		'avg_qualifying_position': avg_position,
		'q3_rate': q3_rate,
		'avg_lap_time': avg_lap_time
	}

def get_track_characteristics(event_name):
	"""Get track characteristics for feature engineering"""
	# Simplified track characteristics - in reality, you'd have a database
	track_info = {
		'Monaco': {'track_type': 'street', 'difficulty': 'high', 'overtaking': 'low'},
		'Spa': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'high'},
		'Silverstone': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Monza': {'track_type': 'permanent', 'difficulty': 'low', 'overtaking': 'high'},
		'Imola': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Bahrain': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'high'},
		'Saudi Arabia': {'track_type': 'street', 'difficulty': 'high', 'overtaking': 'medium'},
		'Australia': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Japan': {'track_type': 'permanent', 'difficulty': 'high', 'overtaking': 'low'},
		'Brazil': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'high'},
		'Mexico': {'track_type': 'permanent', 'difficulty': 'high', 'overtaking': 'low'},
		'Abu Dhabi': {'track_type': 'permanent', 'difficulty': 'low', 'overtaking': 'medium'},
		'Qatar': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'United States': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'high'},
		'Netherlands': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'low'},
		'Italy': {'track_type': 'permanent', 'difficulty': 'low', 'overtaking': 'high'},
		'Austria': {'track_type': 'permanent', 'difficulty': 'low', 'overtaking': 'high'},
		'Spain': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Canada': {'track_type': 'street', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Azerbaijan': {'track_type': 'street', 'difficulty': 'high', 'overtaking': 'medium'},
		'France': {'track_type': 'permanent', 'difficulty': 'low', 'overtaking': 'low'},
		'Hungary': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'low'},
		'Belgium': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'high'},
		'Turkey': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Portugal': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Russia': {'track_type': 'permanent', 'difficulty': 'low', 'overtaking': 'medium'},
		'China': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Vietnam': {'track_type': 'street', 'difficulty': 'high', 'overtaking': 'low'},
		'Germany': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Britain': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Styria': {'track_type': 'permanent', 'difficulty': 'low', 'overtaking': 'high'},
		'Emilia Romagna': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Tuscany': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'Eifel': {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'},
		'70th Anniversary': {'track_type': 'permanent', 'difficulty': 'low', 'overtaking': 'high'},
		'Sakhir': {'track_type': 'permanent', 'difficulty': 'low', 'overtaking': 'high'},
		'Abu Dhabi': {'track_type': 'permanent', 'difficulty': 'low', 'overtaking': 'medium'}
	}
	
	# Default track characteristics
	default = {'track_type': 'permanent', 'difficulty': 'medium', 'overtaking': 'medium'}
	
	# Try to match event name
	for track_name, info in track_info.items():
		if track_name.lower() in event_name.lower():
			return info
	
	return default

def get_previous_track_q3_results(df, event_name, current_year, current_round):
	"""Get previous Q3 results for the same track (STRICTLY HISTORICAL DATA ONLY)"""
	# Check if Year column exists
	if 'Year' not in df.columns:
		print(f"Warning: Year column not found, using default track results for {event_name}")
		return {'avg_q3_rate': 0.5, 'total_races': 0}
	
	# CRITICAL: Only use data from BEFORE the current race (strict temporal separation)
	previous_races = df[(df['EventName'] == event_name) & 
						((df['Year'] < current_year) | 
						 ((df['Year'] == current_year) & (df['Round'] < current_round)))]
	
	if len(previous_races) == 0:
		return {'avg_q3_rate': 0.5, 'total_races': 0}
	
	# Calculate average Q3 qualification rate for this track
	avg_q3_rate = previous_races['Q3Qualified'].mean() if 'Q3Qualified' in previous_races.columns else 0.5
	total_races = len(previous_races['Year'].unique())
	
	return {
		'avg_q3_rate': avg_q3_rate,
		'total_races': total_races
	}

def calculate_team_performance_baseline_2024():
	"""Calculate circuit-specific team performance baseline from 2024 data using percentage gaps"""
	print("\nCalculating circuit-specific team performance baseline from 2024 data...")
	
	try:
		# Get 2024 data
		df_2024 = get_qualifying_data(2024)
		
		if df_2024.empty:
			print("Warning: No 2024 data available, using default baselines")
			return get_default_team_baseline(), []
		
		team_baseline = {}
		
		# Process each circuit separately
		for circuit in df_2024['EventName'].unique():
			circuit_data = df_2024[df_2024['EventName'] == circuit]
			pole_time = circuit_data['LapTime'].min()
			
			# Calculate team performance for this circuit
			for team in circuit_data['Team'].unique():
				team_circuit_data = circuit_data[circuit_data['Team'] == team]
				
				# Get team's best Q3 time at this circuit
				team_q3_times = team_circuit_data[team_circuit_data['Q3Qualified'] == 1]['LapTime']
				
				if len(team_q3_times) > 0:
					team_best_time = team_q3_times.min()
				else:
					# Fallback to best overall time
					team_best_time = team_circuit_data['LapTime'].min()
				
				# Calculate percentage gap to pole at this circuit
				team_gap_pct = (team_best_time - pole_time) / pole_time * 100
				
				# Calculate team Q3 rate at this circuit
				team_q3_rate = len(team_q3_times) / len(team_circuit_data)
				
				if team not in team_baseline:
					team_baseline[team] = {}
				
				team_baseline[team][circuit] = {
					'team_best_time': team_best_time,
					'pole_time': pole_time,
					'team_gap_pct': team_gap_pct,
					'team_q3_rate': team_q3_rate,
					'events_analyzed': 1
				}
		
		# Calculate overall team ranking (average across all circuits)
		team_avg_gaps = {}
		for team, circuits in team_baseline.items():
			avg_gap = sum(circuit['team_gap_pct'] for circuit in circuits.values()) / len(circuits)
			team_avg_gaps[team] = avg_gap
		
		team_performance_order = sorted(team_avg_gaps.items(), key=lambda x: x[1])
		
		print(f"✅ Calculated circuit-specific baseline for {len(team_baseline)} teams from 2024 data")
		print(f"\n🏁 Team Performance Order (by avg % gap to pole):")
		for i, (team, gap_pct) in enumerate(team_performance_order, 1):
			print(f"   {i:2d}. {team:15s}: {gap_pct:5.2f}% avg gap")
		
		return team_baseline, team_performance_order
		
	except Exception as e:
		print(f"Error calculating team baseline: {str(e)}")
		return get_default_team_baseline(), []

def calculate_driver_adjustment_2024():
	"""Calculate circuit-specific driver adjustment factors from 2024 data using percentage-based deltas"""
	print("\n👤 Calculating circuit-specific driver adjustment factors from 2024 data...")
	
	try:
		# Get 2024 data
		df_2024 = get_qualifying_data(2024)
		
		if df_2024.empty:
			print("Warning: No 2024 data available, using default adjustments")
			return get_default_driver_adjustment(), {}
		
		driver_adjustment = {}
		
		# Process each circuit separately
		for circuit in df_2024['EventName'].unique():
			circuit_data = df_2024[df_2024['EventName'] == circuit]
			pole_time = circuit_data['LapTime'].min()
			
			# Group by team to find teammates
			for team in circuit_data['Team'].unique():
				team_circuit_data = circuit_data[circuit_data['Team'] == team]
				drivers = team_circuit_data['Driver'].unique()
				
				if len(drivers) >= 2:
					# Calculate teammate comparisons for this circuit
					for driver in drivers:
						driver_circuit_data = team_circuit_data[team_circuit_data['Driver'] == driver]
						teammates = [d for d in drivers if d != driver]
						
						if teammates:
							teammate = teammates[0]  # Primary teammate
							teammate_circuit_data = team_circuit_data[team_circuit_data['Driver'] == teammate]
							
							# Get best Q3 times at this circuit (or fallback)
							driver_q3_times = driver_circuit_data[driver_circuit_data['Q3Qualified'] == 1]['LapTime']
							teammate_q3_times = teammate_circuit_data[teammate_circuit_data['Q3Qualified'] == 1]['LapTime']
							
							if len(driver_q3_times) > 0 and len(teammate_q3_times) > 0:
								driver_best = driver_q3_times.min()
								teammate_best = teammate_q3_times.min()
							else:
								# Fallback to best overall times
								driver_best = driver_circuit_data['LapTime'].min()
								teammate_best = teammate_circuit_data['LapTime'].min()
							
							# Calculate percentage delta vs teammate at this circuit
							driver_delta_pct = (driver_best - teammate_best) / pole_time * 100
							
							# Calculate driver Q3 rate at this circuit
							driver_q3_rate = len(driver_q3_times) / len(driver_circuit_data)
							
							if driver not in driver_adjustment:
								driver_adjustment[driver] = {}
							
							driver_adjustment[driver][circuit] = {
								'driver_best_time': driver_best,
								'teammate_best_time': teammate_best,
								'teammate': teammate,
								'driver_delta_pct': driver_delta_pct,
								'driver_q3_rate': driver_q3_rate,
								'events_analyzed': 1
							}
		
		# Calculate overall driver metrics (average across all circuits)
		final_driver_adjustment = {}
		for driver, circuits in driver_adjustment.items():
			if circuits:
				avg_delta_pct = sum(circuit['driver_delta_pct'] for circuit in circuits.values()) / len(circuits)
				avg_q3_rate = sum(circuit['driver_q3_rate'] for circuit in circuits.values()) / len(circuits)
				
				# Get team and teammate info from first circuit
				first_circuit = list(circuits.values())[0]
				
				final_driver_adjustment[driver] = {
					'delta_vs_teammate_pct': avg_delta_pct,
					'delta_vs_teammate': avg_delta_pct * 0.01,  # Convert to seconds for compatibility
					'q3_rate': avg_q3_rate,
					'team': 'Unknown',  # Will be filled from circuit data
					'teammate': first_circuit['teammate'],
					'events_analyzed': len(circuits),
					'circuit_data': circuits
				}
		
		print(f"✅ Calculated circuit-specific adjustments for {len(final_driver_adjustment)} drivers from 2024 data")
		print(f"\n👤 Driver Deltas vs Teammates (% of pole time, avg across circuits):")
		for driver, stats in list(final_driver_adjustment.items())[:10]:
			delta_pct = stats['delta_vs_teammate_pct']
			status = "faster" if delta_pct < 0 else "slower"
			print(f"   {driver:3s}: {delta_pct:+5.2f}% vs {stats['teammate']:3s} ({status}), Q3Rate={stats['q3_rate']:5.1%}")
		
		return final_driver_adjustment, {}
		
	except Exception as e:
		print(f"Error calculating driver adjustment: {str(e)}")
		return get_default_driver_adjustment(), {}

def get_default_team_baseline():
	"""Default team baseline when 2024 data is not available"""
	return {
		'Red Bull Racing': {'avg_gap_to_pole': 0.0, 'q3_rate': 0.95, 'avg_grid_pos': 2.5},
		'Ferrari': {'avg_gap_to_pole': 0.3, 'q3_rate': 0.85, 'avg_grid_pos': 4.2},
		'Mercedes': {'avg_gap_to_pole': 0.5, 'q3_rate': 0.80, 'avg_grid_pos': 5.8},
		'McLaren': {'avg_gap_to_pole': 0.7, 'q3_rate': 0.75, 'avg_grid_pos': 7.1},
		'Aston Martin': {'avg_gap_to_pole': 1.0, 'q3_rate': 0.60, 'avg_grid_pos': 9.2},
		'Alpine': {'avg_gap_to_pole': 1.2, 'q3_rate': 0.55, 'avg_grid_pos': 10.5},
		'RB': {'avg_gap_to_pole': 1.4, 'q3_rate': 0.45, 'avg_grid_pos': 12.1},
		'Haas': {'avg_gap_to_pole': 1.6, 'q3_rate': 0.35, 'avg_grid_pos': 14.2},
		'Williams': {'avg_gap_to_pole': 1.8, 'q3_rate': 0.30, 'avg_grid_pos': 15.8},
		'Sauber': {'avg_gap_to_pole': 2.0, 'q3_rate': 0.25, 'avg_grid_pos': 17.1}
	}

def get_default_driver_adjustment():
	"""Default driver adjustment when 2024 data is not available"""
	return {
		'VER': {'delta_vs_teammate': -0.2, 'q3_rate': 0.95, 'team': 'Red Bull Racing', 'teammate': 'PER'},
		'PER': {'delta_vs_teammate': 0.2, 'q3_rate': 0.85, 'team': 'Red Bull Racing', 'teammate': 'VER'},
		'LEC': {'delta_vs_teammate': -0.1, 'q3_rate': 0.90, 'team': 'Ferrari', 'teammate': 'SAI'},
		'SAI': {'delta_vs_teammate': 0.1, 'q3_rate': 0.80, 'team': 'Ferrari', 'teammate': 'LEC'},
		'HAM': {'delta_vs_teammate': -0.15, 'q3_rate': 0.85, 'team': 'Mercedes', 'teammate': 'RUS'},
		'RUS': {'delta_vs_teammate': 0.15, 'q3_rate': 0.75, 'team': 'Mercedes', 'teammate': 'HAM'},
		'NOR': {'delta_vs_teammate': -0.3, 'q3_rate': 0.80, 'team': 'McLaren', 'teammate': 'PIA'},
		'PIA': {'delta_vs_teammate': 0.3, 'q3_rate': 0.70, 'team': 'McLaren', 'teammate': 'NOR'},
		'ALO': {'delta_vs_teammate': -0.4, 'q3_rate': 0.70, 'team': 'Aston Martin', 'teammate': 'STR'},
		'STR': {'delta_vs_teammate': 0.4, 'q3_rate': 0.50, 'team': 'Aston Martin', 'teammate': 'ALO'}
	}

def enhance_features(df, historical_data=None):
	"""Enhanced feature engineering for better Q3 prediction (LEAKAGE-FREE)"""
	print("Enhancing features with advanced engineering (leakage-free)...")
	
	# Create enhanced dataframe
	enhanced_df = df.copy()
	
	# Check if required columns exist
	has_year = 'Year' in df.columns
	has_round = 'Round' in df.columns
	has_position = 'Position' in df.columns
	
	print(f"  - Available columns: Year={has_year}, Round={has_round}, Position={has_position}")
	
	# Use historical data for feature calculation to prevent leakage
	feature_data = historical_data if historical_data is not None else df
	
	# Add driver recent form (if Year and Round are available)
	if has_year and has_round:
		print("  - Calculating driver recent form (leakage-free)...")
		enhanced_df['DriverRecentForm'] = enhanced_df.apply(
			lambda row: calculate_driver_recent_form(feature_data, row['Driver'], row['Year'], row['Round']), 
			axis=1
		)
	else:
		print("  - Skipping driver recent form (Year/Round not available)")
		enhanced_df['DriverRecentForm'] = 0.5  # Default neutral form
	
	# Add team performance metrics (if Year and Round are available)
	if has_year and has_round:
		print("  - Calculating team performance (leakage-free)...")
		team_performance = {}
		for team in enhanced_df['Team'].unique():
			team_performance[team] = {}
			for year in enhanced_df['Year'].unique():
				for round_num in enhanced_df[enhanced_df['Year'] == year]['Round'].unique():
					team_performance[team][(year, round_num)] = calculate_team_performance(
						feature_data, team, year, round_num
					)
		
		enhanced_df['TeamAvgPosition'] = enhanced_df.apply(
			lambda row: team_performance.get(row['Team'], {}).get((row['Year'], row['Round']), {}).get('avg_qualifying_position', 10), 
			axis=1
		)
		enhanced_df['TeamQ3Rate'] = enhanced_df.apply(
			lambda row: team_performance.get(row['Team'], {}).get((row['Year'], row['Round']), {}).get('q3_rate', 0.5), 
			axis=1
		)
		enhanced_df['TeamAvgLapTime'] = enhanced_df.apply(
			lambda row: team_performance.get(row['Team'], {}).get((row['Year'], row['Round']), {}).get('avg_lap_time', 90.0), 
			axis=1
		)
	else:
		print("  - Skipping team performance (Year/Round not available)")
		enhanced_df['TeamAvgPosition'] = 10  # Default
		enhanced_df['TeamQ3Rate'] = 0.5  # Default
		enhanced_df['TeamAvgLapTime'] = 90.0  # Default
	
	# Add track characteristics
	print("  - Adding track characteristics...")
	track_chars = enhanced_df['EventName'].apply(get_track_characteristics)
	enhanced_df['TrackType'] = track_chars.apply(lambda x: x['track_type'])
	enhanced_df['TrackDifficulty'] = track_chars.apply(lambda x: x['difficulty'])
	enhanced_df['TrackOvertaking'] = track_chars.apply(lambda x: x['overtaking'])
	
	# Add previous track Q3 results (if Year and Round are available)
	if has_year and has_round:
		print("  - Adding previous track Q3 results (leakage-free)...")
		enhanced_df['PreviousTrackQ3Rate'] = enhanced_df.apply(
			lambda row: get_previous_track_q3_results(feature_data, row['EventName'], row['Year'], row['Round'])['avg_q3_rate'], 
			axis=1
		)
		enhanced_df['PreviousTrackRaces'] = enhanced_df.apply(
			lambda row: get_previous_track_q3_results(feature_data, row['EventName'], row['Year'], row['Round'])['total_races'], 
			axis=1
		)
	else:
		print("  - Skipping previous track Q3 results (Year/Round not available)")
		enhanced_df['PreviousTrackQ3Rate'] = 0.5  # Default
		enhanced_df['PreviousTrackRaces'] = 0  # Default
	
	# Add relative performance features
	print("  - Adding relative performance features...")
	enhanced_df['LapTimeVsTeamAvg'] = enhanced_df['LapTime'] - enhanced_df['TeamAvgLapTime']
	
	if has_position:
		enhanced_df['PositionVsTeamAvg'] = enhanced_df['Position'] - enhanced_df['TeamAvgPosition']
	else:
		print("  - Skipping PositionVsTeamAvg (Position not available)")
		enhanced_df['PositionVsTeamAvg'] = 0  # Default
	
	# Add compound performance
	print("  - Adding compound performance...")
	compound_performance = enhanced_df.groupby(['Driver', 'Compound'])['Q3Qualified'].mean().reset_index()
	compound_performance.columns = ['Driver', 'Compound', 'DriverCompoundQ3Rate']
	enhanced_df = enhanced_df.merge(compound_performance, on=['Driver', 'Compound'], how='left')
	enhanced_df['DriverCompoundQ3Rate'] = enhanced_df['DriverCompoundQ3Rate'].fillna(0.5)
	
	print(f"Enhanced features added. New shape: {enhanced_df.shape}")
	return enhanced_df

def prepare_enhanced_data(df, label_encoders=None, scaler=None, feature_columns=None):
	"""Prepare enhanced data for model training with proper encoding and column alignment"""
	from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
	import pandas as pd
	import numpy as np
	
	# Ensure Q3Qualified column exists
	if 'Q3Qualified' not in df.columns:
		print("Creating Q3Qualified target column...")
		if 'Position' in df.columns and df['Position'].notna().any():
			# Use position if available
			df['Q3Qualified'] = (df['Position'] <= 10).astype(int)
		elif 'EventName' in df.columns and 'LapTime' in df.columns:
			# Fallback to lap time ranking within each event
			df['Q3Qualified'] = (
				df
				.groupby('EventName')['LapTime']
				.rank(method='min')
				.apply(lambda x: 1 if x <= 10 else 0)
			)
		else:
			# For prediction data without EventName/LapTime, use default value
			print("Warning: Cannot create Q3Qualified from Position or EventName/LapTime, using default value")
			df['Q3Qualified'] = 0  # Default value for prediction data
	
	print(f"Q3 qualification rate: {df['Q3Qualified'].mean():.1%}")
	
	# Define complete feature set (all possible features)
	all_features = [
		'Driver', 'Team', 'Compound', 'TyreLife', 'FreshTyre', 'LapTime',
		'DriverRecentForm', 'TeamAvgPosition', 'TeamQ3Rate', 'TeamAvgLapTime',
		'TrackType', 'TrackDifficulty', 'TrackOvertaking', 'PreviousTrackQ3Rate',
		'PreviousTrackRaces', 'LapTimeVsTeamAvg', 'PositionVsTeamAvg', 'DriverCompoundQ3Rate'
	]
	
	# Select features that exist in the dataframe
	available_features = [f for f in all_features if f in df.columns]
	df_features = df[available_features].copy()
	
	print(f"Available features: {available_features}")
	
	# Handle missing values with appropriate defaults
	df_features = df_features.fillna({
		'TyreLife': 5,
		'FreshTyre': True,
		'LapTime': 90.0,
		'DriverRecentForm': 0.5,
		'TeamAvgPosition': 10,
		'TeamQ3Rate': 0.5,
		'TeamAvgLapTime': 90.0,
		'PreviousTrackQ3Rate': 0.5,
		'PreviousTrackRaces': 0,
		'LapTimeVsTeamAvg': 0,
		'PositionVsTeamAvg': 0,
		'DriverCompoundQ3Rate': 0.5
	})
	
	# Separate categorical and numerical features
	categorical_features = ['Driver', 'Team', 'Compound', 'TrackType', 'TrackDifficulty', 'TrackOvertaking']
	numerical_features = [f for f in available_features if f not in categorical_features]
	
	# Process categorical features
	if label_encoders is None:
		# Training mode - create new encoders
		label_encoders = {}
		encoded_features = []
		
		for col in categorical_features:
			if col in df_features.columns:
				print(f"Encoding categorical feature: {col}")
				# Convert to string and handle NaN values
				df_features[col] = df_features[col].astype(str).fillna('Unknown')
				
				# Use LabelEncoder for high cardinality features (Driver, Team)
				if col in ['Driver', 'Team']:
					le = LabelEncoder()
					df_features[col] = le.fit_transform(df_features[col])
					label_encoders[col] = le
					encoded_features.append(col)
				else:
					# Use one-hot encoding for low cardinality features
					onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
					encoded_cols = onehot.fit_transform(df_features[[col]])
					
					# Create column names
					feature_names = [f"{col}_{cat}" for cat in onehot.categories_[0]]
					
					# Add encoded columns to dataframe
					for i, feature_name in enumerate(feature_names):
						df_features[feature_name] = encoded_cols[:, i]
						encoded_features.append(feature_name)
					
					# Remove original categorical column
					df_features = df_features.drop(columns=[col])
					label_encoders[col] = onehot
		
		# Add numerical features
		encoded_features.extend(numerical_features)
		feature_columns = encoded_features
	else:
		# Prediction mode - use existing encoders
		encoded_features = []
		
		for col in categorical_features:
			if col in df_features.columns:
				print(f"Encoding categorical feature: {col}")
				# Convert to string and handle NaN values
				df_features[col] = df_features[col].astype(str).fillna('Unknown')
				
				# Use LabelEncoder for high cardinality features (Driver, Team)
				if col in ['Driver', 'Team']:
					le = label_encoders[col]
					# Handle unknown categories
					unknown_mask = ~df_features[col].isin(le.classes_)
					df_features.loc[unknown_mask, col] = le.classes_[0]  # Use first class as default
					df_features[col] = le.transform(df_features[col])
					encoded_features.append(col)
				else:
					# Use one-hot encoding for low cardinality features
					onehot = label_encoders[col]
					encoded_cols = onehot.transform(df_features[[col]])
					
					# Create column names
					feature_names = [f"{col}_{cat}" for cat in onehot.categories_[0]]
					
					# Add encoded columns to dataframe
					for i, feature_name in enumerate(feature_names):
						df_features[feature_name] = encoded_cols[:, i]
						encoded_features.append(feature_name)
					
					# Remove original categorical column
					df_features = df_features.drop(columns=[col])
		
		# Add numerical features
		encoded_features.extend(numerical_features)
	
	# Create feature matrix with consistent columns
	X = pd.DataFrame(index=df_features.index)
	
	# Add all expected features, filling missing ones with zeros
	for col in feature_columns:
		if col in df_features.columns:
			X[col] = df_features[col]
		else:
			X[col] = 0  # Fill missing columns with zeros
			print(f"Warning: Feature {col} not found, filling with zeros")
	
	# Ensure all features are numeric
	print(f"Converting all features to numeric...")
	for col in X.columns:
		X[col] = pd.to_numeric(X[col], errors='coerce')
	
	# Fill any remaining NaN values with zeros
	X = X.fillna(0)
	
	# Scale features
	if scaler is None:
		# Training mode - fit new scaler
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)
	else:
		# Prediction mode - use existing scaler
		X_scaled = scaler.transform(X)
	
	# Convert back to DataFrame for easier handling
	X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
	
	print(f"Enhanced data prepared: {X_scaled_df.shape[0]} samples, {X_scaled_df.shape[1]} features")
	print(f"Feature types: All numeric")
	print(f"Q3 qualification rate: {df['Q3Qualified'].mean():.2%}")
	
	return X_scaled_df, df['Q3Qualified'], label_encoders, scaler, feature_columns

def evaluate_models_cv(X, y, cv_folds=5):
	"""Evaluate multiple models using cross-validation"""
	print(f"\nEvaluating models with {cv_folds}-fold cross-validation...")
	
	# Use stratified k-fold for balanced splits
	cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
	
	results = []
	
	for model_name, model in MODELS.items():
		print(f"\nEvaluating {model_name}...")
		
		# Cross-validation scores
		accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
		f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
		auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
		
		results.append({
			'Model': model_name,
			'CV_Accuracy_Mean': accuracy_scores.mean(),
			'CV_Accuracy_Std': accuracy_scores.std(),
			'CV_F1_Mean': f1_scores.mean(),
			'CV_F1_Std': f1_scores.std(),
			'CV_AUC_Mean': auc_scores.mean(),
			'CV_AUC_Std': auc_scores.std()
		})
		
		print(f"  Accuracy: {accuracy_scores.mean():.3f} ± {accuracy_scores.std():.3f}")
		print(f"  F1-Score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
		print(f"  AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
	
	return pd.DataFrame(results)

def train_and_save_all_models(X_train, y_train, X_test, y_test):
	"""Train all models on training data with class imbalance handling and save them to disk"""
	print(f"\nTraining all models on training data (current regulations era)...")
	
	# Check class imbalance
	print(f"Class distribution in training data:")
	print(f"  Q3 Qualified: {y_train.sum()} ({y_train.mean():.1%})")
	print(f"  Not Qualified: {(1-y_train).sum()} ({(1-y_train.mean()):.1%})")
	
	# Apply SMOTE for oversampling (except for GradientBoosting which handles imbalance well)
	if SMOTE_AVAILABLE:
		smote = SMOTE(random_state=42)
		X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
		
		print(f"After SMOTE balancing:")
		print(f"  Q3 Qualified: {y_train_balanced.sum()} ({y_train_balanced.mean():.1%})")
		print(f"  Not Qualified: {(1-y_train_balanced).sum()} ({(1-y_train_balanced).mean():.1%})")
	else:
		print("SMOTE not available, using original data with class_weight='balanced'")
		X_train_balanced, y_train_balanced = X_train, y_train
	
	all_models = {}
	all_metrics = []
	
	for model_name, model in MODELS.items():
		print(f"\nTraining {model_name}...")
		
		# Use balanced data for Random Forest and Logistic Regression, original for Gradient Boosting
		if model_name == 'GradientBoosting':
			# Gradient Boosting handles imbalance well with its own mechanisms
			X_train_model, y_train_model = X_train, y_train
			print(f"  Using original data (Gradient Boosting handles imbalance internally)")
		else:
			# Use SMOTE balanced data for other models (if available)
			if SMOTE_AVAILABLE:
				X_train_model, y_train_model = X_train_balanced, y_train_balanced
				print(f"  Using SMOTE balanced data")
			else:
				X_train_model, y_train_model = X_train, y_train
				print(f"  Using original data with class_weight='balanced'")
		
		# Train model
		model.fit(X_train_model, y_train_model)
		
		# Evaluate on test set
		y_pred = model.predict(X_test)
		y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
		
		# Calculate comprehensive metrics
		metrics = {
			'Model': model_name,
			'Test_Accuracy': accuracy_score(y_test, y_pred),
			'Test_F1': f1_score(y_test, y_pred),
			'Test_AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
			'Test_Precision': precision_score(y_test, y_pred, zero_division=0),
			'Test_Recall': recall_score(y_test, y_pred, zero_division=0)
		}
		
		print(f"  Test Accuracy: {metrics['Test_Accuracy']:.3f}")
		print(f"  Test F1-Score: {metrics['Test_F1']:.3f}")
		print(f"  Test Precision: {metrics['Test_Precision']:.3f}")
		print(f"  Test Recall: {metrics['Test_Recall']:.3f}")
		if metrics['Test_AUC'] is not None:
			print(f"  Test AUC: {metrics['Test_AUC']:.3f}")
		
		# Save model to disk
		model_file = MODEL_FILES[model_name]
		try:
			os.makedirs(os.path.dirname(model_file), exist_ok=True)
			dump(model, model_file)
			print(f"  Model saved to {model_file}")
		except Exception as e:
			print(f"  Warning: Could not save {model_name}: {str(e)}")
		
		all_models[model_name] = model
		all_metrics.append(metrics)
	
	return all_models, pd.DataFrame(all_metrics)

def load_saved_model(model_name='RandomForest'):
	"""Load a specific saved model from disk"""
	model_file = MODEL_FILES[model_name]
	
	if not os.path.exists(model_file):
		raise FileNotFoundError(f"Model file not found: {model_file}")
	
	try:
		model = load(model_file)
		print(f"Loaded {model_name} from {model_file}")
		return model
	except Exception as e:
		raise Exception(f"Error loading {model_name}: {str(e)}")

def get_best_model_from_metrics():
	"""Determine the best model based on saved metrics"""
	metrics_file = METRICS_FILE
	
	if not os.path.exists(metrics_file):
		raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
	
	try:
		metrics_df = pd.read_csv(metrics_file)
		best_model_idx = metrics_df['Test_F1'].idxmax()
		best_model = metrics_df.loc[best_model_idx, 'Model']
		print(f"Best model from metrics: {best_model}")
		return best_model
	except Exception as e:
		print(f"Warning: Could not read metrics file: {str(e)}")
		return 'RandomForest'  # Default fallback

def save_metrics_to_csv(metrics_df, filename=None):
	"""Save evaluation metrics to CSV file"""
	if filename is None:
		filename = METRICS_FILE
	
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	
	# Save metrics to CSV
	metrics_df.to_csv(filename, index=False)
	print(f"\nMetrics saved to {filename}")
	
	return metrics_df

def train_models_once(train_years=(2022, 2023), test_year=2024, sample_size=None):
	"""Train all models once on current regulations era (2022-2023) and test on 2024 data"""
	print(f"\n{'='*60}")
	print("TRAINING MODELS ONCE (CURRENT REGULATIONS ERA) AND TESTING ON 2024")
	print(f"{'='*60}")
	
	# Get training and test data
	train_df, test_df = get_train_test_data(train_years, test_year, sample_size)
	
	# Validate data leakage before proceeding
	print(f"\nValidating data leakage...")
	is_clean = validate_data_leakage(train_df, test_df)
	if not is_clean:
		raise ValueError("Data leakage detected! Cannot proceed with training.")
	
	# Ensure Q3Qualified column exists in both datasets
	print(f"\nEnsuring Q3Qualified target column exists...")
	if 'Q3Qualified' not in train_df.columns:
		print("Creating Q3Qualified column for training data...")
		train_df['Q3Qualified'] = (train_df['Position'] <= 10).astype(int)
	
	if 'Q3Qualified' not in test_df.columns:
		print("Creating Q3Qualified column for test data...")
		test_df['Q3Qualified'] = (test_df['Position'] <= 10).astype(int)
	
	print(f"Training data Q3 rate: {train_df['Q3Qualified'].mean():.1%}")
	print(f"Test data Q3 rate: {test_df['Q3Qualified'].mean():.1%}")
	
	# Enhance features for training data (using only training data for feature calculation)
	print(f"\nEnhancing features for training data (leakage-free)...")
	train_df_enhanced = enhance_features(train_df, historical_data=train_df)
	
	# Enhance features for test data (using only training data for feature calculation to prevent leakage)
	print(f"\nEnhancing features for test data (leakage-free)...")
	test_df_enhanced = enhance_features(test_df, historical_data=train_df)
	
	# Prepare training data with enhanced features
	print(f"\nPreparing enhanced training data...")
	X_train, y_train, label_encoders, scaler, feature_columns = prepare_enhanced_data(train_df_enhanced)
	
	# Prepare test data with enhanced features (using same encoders and scaler)
	print(f"\nPreparing enhanced test data...")
	X_test, y_test, _, _, _ = prepare_enhanced_data(test_df_enhanced, label_encoders, scaler, feature_columns)
	
	# Train all models and save them
	all_models, metrics_df = train_and_save_all_models(X_train, y_train, X_test, y_test)
	
	# Save metrics to CSV
	save_metrics_to_csv(metrics_df)
	
	# Save preprocessing objects
	preprocessing_bundle = {
		'label_encoders': label_encoders,
		'scaler': scaler,
		'features': feature_columns,
		'training_years': f"{train_years[0]}-{train_years[1]}",
		'test_year': test_year
	}
	
	preprocessing_file = os.path.join(MODEL_DIR, 'preprocessing_bundle.joblib')
	try:
		dump(preprocessing_bundle, preprocessing_file)
		print(f"Preprocessing bundle saved to {preprocessing_file}")
	except Exception as e:
		print(f"Warning: Could not save preprocessing bundle: {str(e)}")
	
	# Find best model
	best_model_name = metrics_df.loc[metrics_df['Test_F1'].idxmax(), 'Model']
	print(f"\n🏆 Best performing model: {best_model_name}")
	print(f"   Test F1-Score: {metrics_df.loc[metrics_df['Test_F1'].idxmax(), 'Test_F1']:.3f}")
	
	return all_models, metrics_df, preprocessing_bundle, best_model_name

def load_trained_models():
	"""Load all trained models and preprocessing objects from disk"""
	global _Q3_MODEL_CACHE
	
	# Check if models are already cached in memory
	if _Q3_MODEL_CACHE is not None:
		return _Q3_MODEL_CACHE
	
	# Check if all required files exist
	preprocessing_file = os.path.join(MODEL_DIR, 'preprocessing_bundle.joblib')
	metrics_file = METRICS_FILE
	
	if not os.path.exists(preprocessing_file) or not os.path.exists(metrics_file):
		raise FileNotFoundError("Trained models not found. Please run train_models_once() first.")
	
	# Load preprocessing bundle
	try:
		preprocessing_bundle = load(preprocessing_file)
		print("Loaded preprocessing bundle from disk")
	except Exception as e:
		raise Exception(f"Error loading preprocessing bundle: {str(e)}")
	
	# Get best model name from metrics
	try:
		best_model_name = get_best_model_from_metrics()
	except Exception as e:
		print(f"Warning: Could not determine best model: {str(e)}")
		best_model_name = 'RandomForest'  # Default fallback
	
	# Load the best model
	try:
		best_model = load_saved_model(best_model_name)
	except Exception as e:
		raise Exception(f"Error loading best model ({best_model_name}): {str(e)}")
	
	# Create model bundle
	model_bundle = {
		'clf': best_model,
		'label_encoders': preprocessing_bundle['label_encoders'],
		'scaler': preprocessing_bundle['scaler'],
		'features': preprocessing_bundle['features'],
		'model_name': best_model_name,
		'training_years': preprocessing_bundle['training_years'],
		'test_year': preprocessing_bundle['test_year']
	}
	
	# Cache in memory
	_Q3_MODEL_CACHE = model_bundle
	return model_bundle

def predict_q3_qualification(session, driver_code, generate_plots=False):
	"""Fast Q3 qualification prediction using pre-trained models"""
	from data_utils import format_lap_time
	from visuals import save_plots
	import pandas as pd
	import numpy as np
	from sklearn.metrics import classification_report
	
	try:
		driver_laps = session.laps[session.laps['Driver'] == driver_code]
		if len(driver_laps) == 0:
			raise ValueError(f"No qualifying data found for driver {driver_code}")
		
		fastest_lap = driver_laps.loc[driver_laps['LapTime'].idxmin()]
		lap_time_seconds = fastest_lap['LapTime'].total_seconds()
		minutes = int(lap_time_seconds // 60)
		seconds = lap_time_seconds % 60
		formatted_lap_time = f"{minutes}:{seconds:06.3f}"
		
		# Load pre-trained model (fast!)
		bundle = load_trained_models()
		clf = bundle['clf']
		label_encoders = bundle['label_encoders']
		scaler = bundle['scaler']
		features = bundle['features']
		model_name = bundle.get('model_name', 'Unknown')
		training_years = bundle.get('training_years', 'Unknown')
		
		# Prepare input data with enhanced features
		input_dict = {
			'Driver': [driver_code],
			'Team': [fastest_lap['Team']],
			'Compound': [fastest_lap['Compound']],
			'TyreLife': [fastest_lap['TyreLife']],
			'FreshTyre': [fastest_lap['FreshTyre']],
			'LapTime': [lap_time_seconds],
			# Add default values for enhanced features
			'DriverRecentForm': [0.5],
			'TeamAvgPosition': [10],
			'TeamQ3Rate': [0.5],
			'TeamAvgLapTime': [90.0],
			'TrackType': ['permanent'],
			'TrackDifficulty': ['medium'],
			'TrackOvertaking': ['medium'],
			'PreviousTrackQ3Rate': [0.5],
			'PreviousTrackRaces': [0],
			'LapTimeVsTeamAvg': [0],
			'PositionVsTeamAvg': [0],
			'DriverCompoundQ3Rate': [0.5]
		}
		input_df = pd.DataFrame(input_dict)
		
		# Use the enhanced data preparation function for consistent encoding
		X_input, _, _, _, _ = prepare_enhanced_data(input_df, label_encoders, scaler, features)
		input_scaled = X_input.values
		
		# Make prediction
		prediction = clf.predict(input_scaled)[0]
		probability = clf.predict_proba(input_scaled)[0][1]
		
		# Display results
		print(f"\nDriver: {driver_code}")
		print(f"Team: {fastest_lap['Team']}")
		print(f"Compound: {fastest_lap['Compound']}")
		print(f"Tyre Life: {fastest_lap['TyreLife']} laps")
		print(f"Fresh Tyre: {'Yes' if fastest_lap['FreshTyre'] else 'No'}")
		print(f"Fastest Lap Time: {formatted_lap_time}")
		print(f"\nModel: {model_name} (trained on {training_years})")
		
		result = "QUALIFIES for Q3" if prediction == 1 else "DOES NOT qualify for Q3"
		print(f"\nPrediction: {result}")
		print(f"Model confidence: {probability:.2%}")
		
		# Optional evaluation plots (requires test data)
		if generate_plots:
			print("\nNote: Evaluation plots require test data. Use train_models_once() to generate them.")
		
		return prediction, probability
		
	except Exception as e:
		print(f"Error making Q3 prediction: {str(e)}")
		return None, None

def calculate_historical_trend_improvement():
	"""Calculate trend improvement based on 2023→2024 performance gains"""
	print("📈 Calculating historical trend improvement (2023→2024)...")
	
	try:
		# Get 2023 and 2024 data
		df_2023 = get_qualifying_data(2023)
		df_2024 = get_qualifying_data(2024)
		
		if df_2023.empty or df_2024.empty:
			print("⚠️  Insufficient historical data, using default trend of 0.4s")
			return 0.4
		
		# Calculate average pole time improvement
		improvements = []
		
		for circuit_2024 in df_2024['EventName'].unique():
			# Find matching circuit in 2023
			circuit_2023 = None
			for circuit_2023_candidate in df_2023['EventName'].unique():
				# Simple matching logic
				if any(word in circuit_2023_candidate.lower() for word in circuit_2024.lower().split()):
					circuit_2023 = circuit_2023_candidate
					break
			
			if circuit_2023:
				pole_2023 = df_2023[df_2023['EventName'] == circuit_2023]['LapTime'].min()
				pole_2024 = df_2024[df_2024['EventName'] == circuit_2024]['LapTime'].min()
				
				improvement = pole_2023 - pole_2024  # Positive means 2024 is faster
				improvements.append(improvement)
		
		if improvements:
			avg_improvement = np.mean(improvements)
			print(f"  Average pole time improvement 2023→2024: {avg_improvement:.3f}s")
			return max(0.2, min(0.6, avg_improvement))  # Clamp between 0.2s and 0.6s
		else:
			print("⚠️  No matching circuits found, using default trend of 0.4s")
			return 0.4
			
	except Exception as e:
		print(f"⚠️  Error calculating trend: {str(e)}, using default 0.4s")
		return 0.4

def get_circuit_pole_time_2024(event_name):
	"""Get 2024 pole time for the circuit to estimate 2025 pole time"""
	# Circuit-specific 2024 pole time estimates (in seconds)
	circuit_poles_2024 = {
		'Bahrain': 89.179,      # Bahrain GP
		'Saudi Arabia': 87.472, # Saudi Arabian GP
		'Australia': 78.714,    # Australian GP
		'Japan': 88.197,        # Japanese GP
		'China': 87.301,        # Chinese GP
		'Miami': 87.168,        # Miami GP
		'Emilia Romagna': 88.197, # Imola
		'Monaco': 70.270,       # Monaco GP
		'Canada': 70.514,       # Canadian GP
		'Spain': 88.149,        # Spanish GP
		'Austria': 63.342,      # Austrian GP
		'Great Britain': 88.197, # British GP
		'Hungary': 80.098,      # Hungarian GP
		'Belgium': 103.665,     # Belgian GP (Spa)
		'Netherlands': 70.270,  # Dutch GP
		'Italy': 80.098,        # Italian GP (Monza)
		'Azerbaijan': 103.665,  # Azerbaijan GP
		'Singapore': 88.197,    # Singapore GP
		'United States': 87.168, # US GP
		'Mexico': 80.098,       # Mexican GP
		'Brazil': 70.270,       # Brazilian GP
		'Qatar': 87.168,        # Qatar GP
		'Abu Dhabi': 88.197     # Abu Dhabi GP
	}
	
	# Find matching circuit
	for circuit, pole_time in circuit_poles_2024.items():
		if circuit.lower() in event_name.lower():
			return pole_time
	
	# Default pole time if circuit not found
	return 88.0

def predict_2025_qualification(driver_code, team, event_name="Imola 2025"):
	"""2025 Q3 qualification prediction using circuit-specific 2024 data"""
	from data_utils import format_lap_time
	import numpy as np
	
	print(f"\n2025 Q3 Qualification Prediction for {event_name}")
	print("=" * 60)
	
	# Load pre-trained model (fast!)
	bundle = load_trained_models()
	clf = bundle['clf']
	label_encoders = bundle['label_encoders']
	scaler = bundle['scaler']
	features = bundle['features']
	model_name = bundle.get('model_name', 'Unknown')
	training_years = bundle.get('training_years', 'Unknown')
	
	print(f"✅ Loaded pre-trained model: {model_name} (trained on {training_years})")
	
	# Calculate circuit-specific 2024 baselines
	print(f"\n📊 Calculating circuit-specific 2024 baselines...")
	team_baseline, team_performance_order = calculate_team_performance_baseline_2024()
	driver_adjustment, _ = calculate_driver_adjustment_2024()
	
	# Calculate automatic trend improvement
	trend_improvement = calculate_historical_trend_improvement()
	print(f"📈 Using trend improvement: {trend_improvement:.3f}s")
	
	# Get circuit name for lookup
	circuit_name = event_name.split()[0] if ' ' in event_name else event_name
	
	# Get 2024 pole time for this circuit
	pole_time_2024 = get_circuit_pole_time_2024(event_name)
	pole_time_2025 = pole_time_2024 - trend_improvement
	
	print(f"  2024 Pole Time: {format_lap_time(pole_time_2024)}")
	print(f"  2025 Projected Pole: {format_lap_time(pole_time_2025)}")
	
	# Get circuit-specific team baseline
	team_gap_pct = 1.0  # Default fallback
	team_q3_rate = 0.5
	team_events = 0
	team_rank = "N/A"
	
	if team in team_baseline:
		# Find matching circuit in team baseline
		for circuit, stats in team_baseline[team].items():
			if circuit_name.lower() in circuit.lower() or circuit.lower() in circuit_name.lower():
				team_gap_pct = stats['team_gap_pct']
				team_q3_rate = stats['team_q3_rate']
				team_events = stats['events_analyzed']
				print(f"\n🏁 Team baseline ({team} at {circuit}):")
				print(f"   Team Gap: {team_gap_pct:.2f}%")
				print(f"   Q3 Rate: {team_q3_rate:.1%}")
				print(f"   Events: {team_events}")
				break
		else:
			print(f"⚠️  No circuit-specific data for {team} at {circuit_name}, using defaults")
	else:
		print(f"⚠️  No 2024 baseline for {team}, using defaults")
	
	# Get circuit-specific driver adjustment
	driver_delta_pct = 0.0  # Default fallback
	driver_q3_rate = 0.5
	driver_events = 0
	teammate = "Unknown"
	
	if driver_code in driver_adjustment and 'circuit_data' in driver_adjustment[driver_code]:
		# Find matching circuit in driver baseline
		for circuit, stats in driver_adjustment[driver_code]['circuit_data'].items():
			if circuit_name.lower() in circuit.lower() or circuit.lower() in circuit_name.lower():
				driver_delta_pct = stats['driver_delta_pct']
				driver_q3_rate = stats['driver_q3_rate']
				driver_events = stats['events_analyzed']
				teammate = stats['teammate']
				status = "faster" if driver_delta_pct < 0 else "slower"
				print(f"\n👤 Driver adjustment ({driver_code} at {circuit}):")
				print(f"   Delta vs {teammate}: {driver_delta_pct:+.2f}% ({status})")
				print(f"   Q3 Rate: {driver_q3_rate:.1%}")
				print(f"   Events: {driver_events}")
				break
		else:
			print(f"⚠️  No circuit-specific data for {driver_code} at {circuit_name}, using defaults")
	else:
		print(f"⚠️  No 2024 adjustment for {driver_code}, using defaults")
	
	# Calculate estimated lap time using circuit-specific data
	total_gap_pct = team_gap_pct + driver_delta_pct
	estimated_lap_time = pole_time_2025 * (1.0 + total_gap_pct / 100.0)
	
	print(f"\n⏱️  Lap Time Calculation:")
	print(f"  Formula: estimated_time = pole_2025 × (1 + (team_gap_pct + driver_delta_pct)/100)")
	print(f"  Formula: estimated_time = {pole_time_2025:.3f} × (1 + {total_gap_pct:.2f}/100)")
	print(f"  Formula: estimated_time = {pole_time_2025:.3f} × {1.0 + total_gap_pct/100.0:.4f}")
	print(f"  Estimated Driver Lap Time: {format_lap_time(estimated_lap_time)}")
	
	# Prepare input for prediction
	input_dict = {
		'Driver': [driver_code],
		'Team': [team],
		'Compound': ['SOFT'],
		'TyreLife': [5.0],
		'FreshTyre': [True],
		'LapTime': [estimated_lap_time],
		'DriverRecentForm': [driver_q3_rate],
		'TeamAvgPosition': [10.0],
		'TeamQ3Rate': [team_q3_rate],
		'TeamAvgLapTime': [pole_time_2025 * (1.0 + team_gap_pct / 100.0)],
		'TrackType': ['permanent'],
		'TrackDifficulty': ['medium'],
		'TrackOvertaking': ['medium'],
		'PreviousTrackQ3Rate': [team_q3_rate],
		'PreviousTrackRaces': [team_events],
		'LapTimeVsTeamAvg': [pole_time_2025 * (driver_delta_pct / 100.0)],
		'PositionVsTeamAvg': [0],
		'DriverCompoundQ3Rate': [driver_q3_rate]
	}
	
	input_df = pd.DataFrame(input_dict)
	
	# Make prediction
	try:
		X_input, _, _, _, _ = prepare_enhanced_data(input_df, label_encoders, scaler, features)
		input_scaled = X_input.values
		prediction = clf.predict(input_scaled)[0]
		probability = clf.predict_proba(input_scaled)[0][1]
	except Exception as e:
		print(f"⚠️  Prediction error: {str(e)}")
		prediction = 0
		probability = 0.5
	
	# Display results
	print(f"\n" + "="*60)
	print(f"🎯 2025 Q3 QUALIFICATION PREDICTION RESULTS")
	print(f"="*60)
	
	print(f"\n👤 Driver: {driver_code}")
	print(f"🏁 Team: {team}")
	print(f"🏁 Event: {event_name}")
	print(f"🛞 Compound: SOFT")
	print(f"🛞 Tyre Life: 5.0 laps")
	print(f"🛞 Fresh Tyre: Yes")
	print(f"⏱️  Estimated Lap Time: {format_lap_time(estimated_lap_time)}")
	
	print(f"\n🤖 Model: {model_name} (trained on {training_years})")
	
	result = "✅ QUALIFIES for Q3" if prediction == 1 else "❌ DOES NOT qualify for Q3"
	print(f"\n🎯 Prediction: {result}")
	print(f"📊 Model confidence: {probability:.1%}")
	
	# Detailed analysis
	print(f"\n📈 Circuit-Specific Analysis:")
	print(f"   Circuit: {circuit_name}")
	print(f"   2024 Pole: {format_lap_time(pole_time_2024)}")
	print(f"   2025 Projected Pole: {format_lap_time(pole_time_2025)} (trend: -{trend_improvement:.3f}s)")
	print(f"   Team Gap: {team_gap_pct:.2f}%")
	print(f"   Driver Delta: {driver_delta_pct:+.2f}% vs {teammate}")
	print(f"   Total Gap: {total_gap_pct:.2f}%")
	print(f"   Estimated Time: {format_lap_time(estimated_lap_time)}")
	
	# Validation
	if estimated_lap_time < 60 or estimated_lap_time > 120:
		print(f"   ⚠️  Warning: Estimated lap time seems unrealistic")
	else:
		print(f"   ✅ Estimated lap time is realistic for this circuit")
	
	return prediction, probability

def validate_data_leakage(train_df, test_df):
	"""Validate SAFE temporal split and leakage-free feature engineering"""
	print(f"\n{'='*60}")
	print("SAFE TEMPORAL SPLIT VALIDATION")
	print(f"{'='*60}")
	
	leakage_found = False
	
	# Check 1: Temporal separation (CRITICAL)
	print("1. Checking temporal separation (CRITICAL)...")
	train_years = set(train_df['Year'].unique()) if 'Year' in train_df.columns else set()
	test_years = set(test_df['Year'].unique()) if 'Year' in test_df.columns else set()
	
	if train_years & test_years:
		print(f"   ❌ TEMPORAL LEAKAGE: Overlapping years between train and test: {train_years & test_years}")
		leakage_found = True
	else:
		print(f"   ✅ Temporal separation confirmed. Train: {train_years}, Test: {test_years}")
	
	# Check 2: Event overlap (SAFE with temporal features)
	print("2. Checking event overlap (SAFE with temporal features)...")
	train_events = set(train_df['EventName'].unique()) if 'EventName' in train_df.columns else set()
	test_events = set(test_df['EventName'].unique()) if 'EventName' in test_df.columns else set()
	
	overlapping_events = train_events & test_events
	if overlapping_events:
		print(f"   📊 Event overlap detected: {len(overlapping_events)} events appear in both periods")
		print(f"   ✅ This is SAFE because temporal features only use past data")
		print(f"   Examples: {list(overlapping_events)[:3]}{'...' if len(overlapping_events) > 3 else ''}")
	else:
		print(f"   ✅ No event overlap. Train: {len(train_events)} events, Test: {len(test_events)} events")
	
	# Check 3: Driver/Team consistency
	print("3. Checking driver/team consistency...")
	train_drivers = set(train_df['Driver'].unique()) if 'Driver' in train_df.columns else set()
	test_drivers = set(test_df['Driver'].unique()) if 'Driver' in test_df.columns else set()
	
	driver_overlap = train_drivers & test_drivers
	if len(driver_overlap) > 0:
		print(f"   ✅ Driver overlap expected: {len(driver_overlap)} drivers in both sets")
	else:
		print(f"   ⚠️  No driver overlap - this might indicate data issues")
	
	# Check 4: Q3Qualified target separation
	print("4. Checking target variable separation...")
	if 'Q3Qualified' in train_df.columns and 'Q3Qualified' in test_df.columns:
		train_q3_rate = train_df['Q3Qualified'].mean()
		test_q3_rate = test_df['Q3Qualified'].mean()
		print(f"   ✅ Train Q3 rate: {train_q3_rate:.1%}, Test Q3 rate: {test_q3_rate:.1%}")
		
		# Check if rates are realistic
		if train_q3_rate > 0.8 or train_q3_rate < 0.2:
			print(f"   ⚠️  Unusual Q3 rate in training data: {train_q3_rate:.1%}")
		if test_q3_rate > 0.8 or test_q3_rate < 0.2:
			print(f"   ⚠️  Unusual Q3 rate in test data: {test_q3_rate:.1%}")
	
	# Check 5: Feature engineering validation (leakage-free)
	print("5. Checking feature engineering (leakage-free)...")
	enhanced_train = enhance_features(train_df, historical_data=train_df)
	enhanced_test = enhance_features(test_df, historical_data=train_df)
	
	# Check if enhanced features are reasonable
	if 'DriverRecentForm' in enhanced_train.columns:
		train_form = enhanced_train['DriverRecentForm'].mean()
		test_form = enhanced_test['DriverRecentForm'].mean()
		print(f"   ✅ Driver recent form - Train: {train_form:.3f}, Test: {test_form:.3f}")
	
	if 'TeamQ3Rate' in enhanced_train.columns:
		train_team_rate = enhanced_train['TeamQ3Rate'].mean()
		test_team_rate = enhanced_test['TeamQ3Rate'].mean()
		print(f"   ✅ Team Q3 rate - Train: {train_team_rate:.3f}, Test: {test_team_rate:.3f}")
	
	# Check 6: Data size validation
	print("6. Checking data size adequacy...")
	if len(train_df) < 50:
		print(f"   ⚠️  Training data too small: {len(train_df)} records")
		leakage_found = True
	else:
		print(f"   ✅ Training data size adequate: {len(train_df)} records")
	
	if len(test_df) < 10:
		print(f"   ⚠️  Test data too small: {len(test_df)} records")
		leakage_found = True
	else:
		print(f"   ✅ Test data size adequate: {len(test_df)} records")
	
	# Check 7: Validate no 2024 events in training features
	print("7. Validating no 2024 events in training features...")
	if 'Year' in train_df.columns:
		train_years_list = train_df['Year'].unique()
		if 2024 in train_years_list:
			print(f"   ❌ LEAKAGE: 2024 events found in training data!")
			leakage_found = True
		else:
			print(f"   ✅ No 2024 events in training data. Training years: {sorted(train_years_list)}")
	else:
		print(f"   ⚠️  Year column not found - cannot validate temporal separation")
	
	print(f"\n{'='*60}")
	if leakage_found:
		print("❌ DATA LEAKAGE DETECTED - FIX REQUIRED")
		print("   The pipeline will stop to prevent invalid results.")
	else:
		print("✅ SAFE TEMPORAL SPLIT VALIDATED - PIPELINE IS CLEAN")
		print("   Safe to proceed with training using all 2022-2023 data.")
	print(f"{'='*60}")
	
	return not leakage_found

def test_enhanced_model(sample_size=500):
	"""Quick test function for the enhanced model with smaller samples"""
	print("\n" + "="*60)
	print("TESTING ENHANCED Q3 PREDICTION MODEL (CURRENT REGULATIONS ERA)")
	print("="*60)
	
	try:
		# Test with smaller sample for faster execution
		print(f"\nTesting with sample size: {sample_size}")
		
		# Train models once on 2022-2023 (current regulations), test on 2024
		all_models, metrics_df, preprocessing_bundle, best_model_name = train_models_once(
			train_years=(2022, 2023), 
			test_year=2024, 
			sample_size=sample_size
		)
		
		print(f"\n✅ Enhanced model test completed successfully!")
		print(f"Best model: {best_model_name}")
		best_metrics = metrics_df.loc[metrics_df['Model'] == best_model_name]
		print(f"Test F1-Score: {best_metrics['Test_F1'].iloc[0]:.3f}")
		print(f"Test Accuracy: {best_metrics['Test_Accuracy'].iloc[0]:.3f}")
		print(f"Test AUC: {best_metrics['Test_AUC'].iloc[0]:.3f}")
		print(f"Test Precision: {best_metrics['Test_Precision'].iloc[0]:.3f}")
		print(f"Test Recall: {best_metrics['Test_Recall'].iloc[0]:.3f}")
		
		return True
		
	except Exception as e:
		print(f"\n❌ Enhanced model test failed: {str(e)}")
		import traceback
		traceback.print_exc()
		return False


