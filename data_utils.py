import os
import pandas as pd
import fastf1 as ff1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ff1.Cache.enable_cache('f1_cache')

def get_qualifying_data(year=2024):
	"Fetch qualifying data for the specified year with proper Q3 qualification target"
	sessions = []
	schedule = ff1.get_event_schedule(year)
	for _, event in schedule.iterrows():
		try:
			session = ff1.get_session(year, event['EventName'], 'Q')
			session.load()
			
			qualifying_results = session.results
			if qualifying_results is None or len(qualifying_results) == 0:
				print(f"No qualifying results for {event['EventName']}, skipping...")
				continue
			
			laps = session.laps
			drivers = session.drivers
			
			session_data = []
			for _, result in qualifying_results.iterrows():
				driver = result['Abbreviation']
				position = result['Position']
				
				# Get best lap data for this driver
				driver_laps = laps[laps['DriverNumber'] == result['DriverNumber']]
				if len(driver_laps) > 0:
					best_lap = driver_laps.loc[driver_laps['LapTime'].idxmin()]
					
					session_data.append({
						'Year': year,
						'EventName': event['EventName'],
						'Round': event['RoundNumber'],
						'Driver': driver,
						'Team': result['TeamName'],
						'Compound': best_lap['Compound'],
						'TyreLife': best_lap['TyreLife'],
						'FreshTyre': best_lap['FreshTyre'],
						'LapTime': best_lap['LapTime'].total_seconds(),
						'Position': position,
						'Q3Qualified': 1 if position <= 10 else 0  # Q3 qualification target
					})
			
			sessions.extend(session_data)
			print(f"Processed {event['EventName']} - {len(session_data)} drivers, Q3 qualified: {sum(1 for d in session_data if d['Q3Qualified'] == 1)}")
		except Exception as e:
			print(f"Error processing {event['EventName']}: {str(e)}")
			continue
	
	if not sessions:
		raise ValueError("No qualifying data was collected. Please check the error messages above.")
	
	df = pd.DataFrame(sessions)
	print(f"\nTotal data collected: {len(df)} records")
	print(f"Q3 qualification rate: {df['Q3Qualified'].mean():.1%}")
	return df

def prepare_data(df):
	"""Preparing the data for model training"""
	print("\nColumns in the dataset:")
	print(df.columns.tolist())
	print(df.head())
	if df.empty:
		raise ValueError("No data available for processing")
	
	# Ensure Q3Qualified column exists
	if 'Q3Qualified' not in df.columns:
		print("Warning: Q3Qualified column not found, creating from Position...")
		if 'Position' in df.columns and df['Position'].notna().any():
			df['Q3Qualified'] = (df['Position'] <= 10).astype(int)
		else:
			# Fallback to lap time ranking within each event
			df['Q3Qualified'] = (
				df
				.groupby('EventName')['LapTime']
				.rank(method='min')
				.apply(lambda x: 1 if x <= 10 else 0)
			)
	
	print(f"Q3 qualification rate: {df['Q3Qualified'].mean():.1%}")
	
	features = ['Driver', 'Team', 'Compound', 'TyreLife', 'FreshTyre', 'LapTime']
	X = df[features]
	y = df['Q3Qualified']
	label_encoders = {}
	for col in ['Driver', 'Team', 'Compound']:
		le = LabelEncoder()
		X[col] = le.fit_transform(X[col].astype(str))
		label_encoders[col] = le
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	return X_scaled, y, label_encoders, scaler, features

def format_lap_time(seconds):
	"""Converting seconds to MM:SS.sss format"""
	minutes = int(seconds // 60)
	remaining_seconds = seconds % 60
	return f"{minutes}:{remaining_seconds:06.3f}"

def get_race_qualifying_data(event_name, driver_code):
	"""Fetching qualifying data for a specific race and driver (2024 season context)"""
	try:
		session = ff1.get_session(2024, event_name, 'Q')
		session.load()
		laps = session.laps
		driver_laps = laps[laps['Driver'] == driver_code]
		if len(driver_laps) == 0:
			raise ValueError(f"No qualifying data found for driver {driver_code} in {event_name}")
		fastest_lap = driver_laps.loc[driver_laps['LapTime'].idxmin()]
		return {
			'Driver': driver_code,
			'Team': fastest_lap['Team'],
			'Compound': fastest_lap['Compound'],
			'TyreLife': fastest_lap['TyreLife'],
			'FreshTyre': fastest_lap['FreshTyre'],
			'LapTime': fastest_lap['LapTime'].total_seconds()
		}
	except Exception as e:
		raise ValueError(f"Error fetching qualifying data: {str(e)}")

def get_race_session(year, event_name, session_type='Q'):
	try:
		print(f"Loading {event_name} {session_type} data")
		session = ff1.get_session(year, event_name, session_type)
		print(f"Loading {session_type} session data")
		if session_type.upper() == 'R':
			print("For race, loading full data including laps, telemetry, and weather")
			session.load(laps=True, telemetry=True, weather=True)
		else:
			print("For qualifying, loading basic data")
			session.load()
		return session
	except Exception as e:
		raise ValueError(f"Error loading session data: {str(e)}")


