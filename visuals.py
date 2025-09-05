import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def save_plots(clf, X_test, y_test, X, driver_name, plots_dir='plots'):
	"""Save evaluation plots (confusion matrix, feature importances, ROC)"""
	if not os.path.exists(plots_dir):
		os.makedirs(plots_dir)
	# Confusion Matrix
	plt.figure(figsize=(10, 8))
	y_pred = clf.predict(X_test)
	conf_matrix = confusion_matrix(y_test, y_pred)
	sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
				xticklabels=['Not Q3', 'Q3'],
				yticklabels=['Not Q3', 'Q3'])
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.title(f"Confusion Matrix - Q3 Qualification Prediction ({driver_name})")
	plt.tight_layout()
	plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{driver_name}.png'), dpi=300, bbox_inches='tight')
	print(f"Confusion matrix saved as '{plots_dir}/confusion_matrix_{driver_name}.png'")
	# Feature Importances
	importances = clf.feature_importances_
	feature_names = X.columns if hasattr(X, 'columns') else [f'F{i}' for i in range(len(importances))]
	feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
	plt.figure(figsize=(12, 8))
	sns.barplot(data=feature_df, x='Importance', y='Feature', palette='viridis')
	plt.title(f"Feature Importances for Q3 Qualification Prediction ({driver_name})")
	plt.xlabel("Importance")
	plt.ylabel("Feature")
	plt.tight_layout()
	plt.savefig(os.path.join(plots_dir, f'feature_importances_{driver_name}.png'), dpi=300, bbox_inches='tight')
	print(f"Feature importances plot saved as '{plots_dir}/feature_importances_{driver_name}.png'")
	# ROC Curve
	y_probs = clf.predict_proba(X_test)[:, 1]
	fpr, tpr, thresholds = roc_curve(y_test, y_probs)
	roc_auc = auc(fpr, tpr)
	plt.figure(figsize=(10, 8))
	plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(f'ROC Curve - Q3 Qualification Prediction ({driver_name})')
	plt.legend(loc="lower right")
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(plots_dir, f'roc_curve_{driver_name}.png'), dpi=300, bbox_inches='tight')
	print(f"ROC curve saved as '{plots_dir}/roc_curve_{driver_name}.png'")

def format_time_axis(x, pos):
	minutes = int(x // 60)
	seconds = x % 60
	return f"{minutes}:{seconds:06.3f}"

def show_race_results(session):
	results = session.results
	plt.style.use('dark_background')
	plt.figure(figsize=(14, 10))
	teams = {}
	for team, color in zip(results['TeamName'], results['TeamColor']):
		# Fallback to gray if color invalid/missing
		try:
			hex_str = (color or '').lstrip('#')
			if len(hex_str) != 6:
				raise ValueError('invalid hex length')
			rgb = tuple(int(hex_str[i:i+2], 16) / 255 for i in (0, 2, 4))
		except Exception:
			rgb = (0.5, 0.5, 0.5)
		teams[team] = rgb
	points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
	points = [points_system.get(pos, 0) for pos in results['Position']]
	y_pos = np.arange(len(results['Abbreviation']))
	plt.barh(y_pos, results['Position'], color=[teams[team] for team in results['TeamName']])
	for i, (abbr, pts) in enumerate(zip(results['Abbreviation'], points)):
		plt.text(0.5, i, f"{abbr}", ha='center', va='center', fontweight='bold', fontsize=12, color='white')
		plt.text(results['Position'][i] + 0.3, i, f"{pts} pts", ha='left', va='center', fontsize=11, color='gold' if pts > 0 else 'gray')
	plt.title(f"Race Results - {session.event['EventName']}", fontsize=16)
	plt.xlabel("Position")
	plt.xlim(0, max(results['Position']) + 3)
	plt.grid(axis='x', alpha=0.3)
	unique_teams = results[['TeamName', 'TeamColor']].drop_duplicates()
	legend_elements = []
	for _, row in unique_teams.iterrows():
		team_name = row['TeamName']
		try:
			hex_str = (row['TeamColor'] or '').lstrip('#')
			if len(hex_str) != 6:
				raise ValueError('invalid hex length')
			rgb = tuple(int(hex_str[i:i+2], 16) / 255 for i in (0, 2, 4))
		except Exception:
			rgb = (0.5, 0.5, 0.5)
		legend_elements.append(plt.Rectangle((0,0), 1, 1, color=rgb, label=team_name))
	plt.legend(handles=legend_elements, title="Teams", loc="lower right")
	plt.tight_layout()
	plt.show()
	plt.style.use('default')

def compare_drivers(session, driver1, driver2):
	try:
		available_drivers = sorted(session.drivers)
		print("\nAvailable drivers:", ', '.join(available_drivers))
		driver1_laps = session.laps[session.laps['Driver'] == driver1]
		driver2_laps = session.laps[session.laps['Driver'] == driver2]
		if driver1_laps.empty or driver2_laps.empty:
			print(f"No lap data available for one or both drivers: {driver1}, {driver2}")
			return
		plt.style.use('dark_background')
		plt.figure(figsize=(15, 8))
		all_laps = session.laps['LapNumber'].unique()
		max_lap = int(max(all_laps)) if len(all_laps) > 0 else 0
		d1_times = driver1_laps['LapTime'].dt.total_seconds()
		d2_times = driver2_laps['LapTime'].dt.total_seconds()
		def hex_to_rgb(hex_color):
			hex_color = hex_color.lstrip('#')
			return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
		color1 = hex_to_rgb(session.get_driver(driver1)['TeamColor'])
		color2 = hex_to_rgb(session.get_driver(driver2)['TeamColor'])
		plt.plot(driver1_laps['LapNumber'], d1_times, label=f"{driver1} ({session.get_driver(driver1)['TeamName']})", color=color1, marker='o', markersize=5, linewidth=2)
		plt.plot(driver2_laps['LapNumber'], d2_times, label=f"{driver2} ({session.get_driver(driver2)['TeamName']})", color=color2, marker='o', markersize=5, linewidth=2)
		avg1 = d1_times.mean(); avg2 = d2_times.mean()
		plt.axhline(y=avg1, color=color1, linestyle='--', alpha=0.5, label=f'{driver1} Avg: {format_time_axis(avg1, None)}')
		plt.axhline(y=avg2, color=color2, linestyle='--', alpha=0.5, label=f'{driver2} Avg: {format_time_axis(avg2, None)}')
		plt.xlim(0, max_lap + 1)
		plt.xticks(range(0, max_lap + 1, 5 if max_lap > 20 else 1))
		plt.title(f'Lap Time Comparison: {driver1} vs {driver2}', fontsize=16)
		plt.xlabel('Lap Number', fontsize=12)
		plt.ylabel('Lap Time (MM:SS.mmm)', fontsize=12)
		plt.legend(fontsize=12, title="Drivers")
		plt.grid(True, alpha=0.3)
		plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_time_axis))
		plt.tight_layout(); plt.show(); plt.style.use('default')
	except Exception as e:
		print(f"Error comparing drivers: {str(e)}")
		import traceback; traceback.print_exc()

def show_tire_strategies(session):
	try:
		plt.style.use('dark_background')
		compound_colors = {'SOFT': '#FF3333','MEDIUM': '#FFF200','HARD': '#FFFFFF','INTERMEDIATE': '#00FF00','WET': '#0000FF'}
		laps = session.laps
		if laps.empty:
			print("\nNo lap data available for this session."); return
		print("\nExtracting tire strategy data")
		stints_data = []
		if session.name.upper() in ['R', 'RACE']:
			print("Using race finishing order for drivers")
			final_positions = {}
			for driver in laps['Driver'].unique():
				driver_laps = laps[laps['Driver'] == driver]
				if not driver_laps.empty:
					last_lap = driver_laps.iloc[-1]
					if 'Position' in last_lap:
						final_positions[driver] = last_lap['Position']
			final_order = [driver for driver, _ in sorted(final_positions.items(), key=lambda x: x[1])]
			if not final_order:
				print("No position data found, using fastest lap order")
				fastest_laps = laps.pick_fastest(); final_order = fastest_laps['Driver'].tolist()
		else:
			print("Using fastest lap order for drivers")
			fastest_laps = laps.pick_fastest(); final_order = fastest_laps['Driver'].tolist()
		print(f"Processing stints for {len(final_order)} drivers in order")
		for driver in final_order:
			driver_laps = laps[laps['Driver'] == driver]
			if not driver_laps.empty:
				if 'Stint' in driver_laps.columns and not driver_laps['Stint'].isna().all():
					try:
						driver_stints = driver_laps.groupby('Stint')['Compound'].first()
						stint_lengths = driver_laps.groupby('Stint').size()
						for stint, compound in driver_stints.items():
							stint_laps = driver_laps[driver_laps['Stint'] == stint]
							if not stint_laps.empty:
								stints_data.append({'Driver': driver,'Stint': stint,'Compound': compound,'Length': stint_lengths[stint],'Start': stint_laps['LapNumber'].min(),'End': stint_laps['LapNumber'].max()})
					except Exception as e:
						print(f"Error processing stints for driver {driver}: {str(e)}")
		if not stints_data:
			print("\nNo tire strategy data available for this session.")
			print("\nAvailable columns:", ', '.join(laps.columns))
			if 'Stint' in laps.columns:
				print("Stint column exists but may be empty or NaN")
			if 'Compound' in laps.columns:
				print("Compound column exists but may be empty or NaN")
			return
		stints_df = pd.DataFrame(stints_data)
		print(f"Successfully extracted {len(stints_df)} stint records.")
		plt.figure(figsize=(15, 10))
		drivers = stints_df['Driver'].unique()
		for idx, driver in enumerate(reversed(drivers)):
			driver_stints = stints_df[stints_df['Driver'] == driver]
			for _, stint in driver_stints.iterrows():
				plt.barh(y=idx, width=stint['Length'], left=stint['Start'] - 1, color=compound_colors.get(stint['Compound'], '#CCCCCC'), edgecolor='black', alpha=0.7)
				if stint['Length'] > 5:
					plt.text(stint['Start'] + stint['Length']/2 - 1, idx, stint['Compound'], ha='center', va='center', fontsize=8)
		session_type_name = {'R': 'Race', 'Q': 'Qualifying', 'S': 'Sprint', 'FP1': 'Practice 1', 'FP2': 'Practice 2', 'FP3': 'Practice 3'}
		session_type = session.name.upper()
		event_name = session.event['EventName']
		year = session.event.year
		plt.title(f"{year} {event_name} GP - {session_type_name.get(session_type, session_type)} Tire Strategies", fontsize=16, pad=20)
		plt.xlabel('Lap Number'); plt.ylabel('Driver (Finishing Order)')
		plt.yticks(range(len(drivers)), reversed(drivers))
		plt.grid(True, alpha=0.3)
		used_compounds = stints_df['Compound'].unique()
		legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=compound_colors.get(c, '#CCCCCC'), edgecolor='black', alpha=0.7, label=c) for c in used_compounds]
		plt.legend(handles=legend_elements, title='Tire Compounds', bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.tight_layout(); plt.show(); plt.style.use('default')
	except Exception as e:
		print(f"Error showing tire strategies: {str(e)}"); import traceback; traceback.print_exc()

def show_session_info(session):
	print("\nSession Information:")
	print(f"Event: {session.event['EventName']}")
	print(f"Date: {session.date}")
	print(f"Track: {session.event['Location']}")
	print(f"Session Type: {session.name}")
	if 'Q' in session.name:
		print("\nQualifying Results:"); print("=" * 80)
		print(f"{'Pos':<5} {'Driver':<10} {'Team':<20} {'Q1':<10} {'Q2':<10} {'Q3':<10}"); print("-" * 80)
		results = session.results
		if not results.empty:
			for _, driver in results.iterrows():
				pos = driver['Position']; driver_code = driver['Abbreviation']; team = driver['TeamName']
				q1 = driver.get('Q1', 'N/A'); q2 = driver.get('Q2', 'N/A'); q3 = driver.get('Q3', 'N/A')
				def fmt(x):
					import pandas as pd
					return format_time_axis(x.total_seconds(), None) if isinstance(x, pd.Timedelta) and not pd.isna(x) else 'N/A'
				print(f"{pos:<5} {driver_code:<10} {team:<20} {fmt(q1):<10} {fmt(q2):<10} {fmt(q3):<10}")
		else:
			print("No qualifying results available")
	print("\nWeather Data:")
	try:
		print(session.weather_data.describe())
	except Exception as e:
		print(f"Weather data not available: {str(e)}")
	print("\nTrack Status:")
	try:
		print(session.track_status)
	except Exception as e:
		print(f"Track status not available: {str(e)}")


