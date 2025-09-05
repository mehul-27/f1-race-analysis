from data_utils import get_race_session
from model_utils import predict_q3_qualification, predict_2025_qualification, train_models_once
from visuals import show_race_results, compare_drivers, show_tire_strategies, show_session_info
import fastf1 as ff1

def analyze_current_season():
	print("\nF1 Race Analysis")
	try:
		year_input = input("\nEnter season year (e.g., 2021-2025): ").strip()
		selected_year = int(year_input)
	except Exception:
		print("Invalid year. Defaulting to 2024.")
		selected_year = 2024
	print(f"\nAvailable races for {selected_year}:")
	schedule = ff1.get_event_schedule(selected_year)
	for idx, event in schedule.iterrows():
		print(f"{idx + 1}. {event['EventName']}")
	race_idx = int(input("\nSelect a race (enter number): ").strip()) - 1
	if race_idx < 0 or race_idx >= len(schedule):
		print("Invalid race number"); return
	event_name = schedule.iloc[race_idx]['EventName']
	print("\nLoading qualifying session")
	session = get_race_session(selected_year, event_name, 'Q')
	race_session = None
	while True:
		print("\nAnalysis Options:")
		print("1. Show Race Results")
		print("2. Compare Two Drivers")
		print("3. Show Tire Strategies")
		print("4. View Session Information")
		print("5. Q3 Qualification Prediction")
		print("6. 2025 Q3 Prediction")
		print("7. Train Models")
		print("8. Exit")
		choice = input("\nSelect an option (1-8): ").strip()
		if choice == '1':
			if race_session is None:
				print("\nLoading race session data")
				try:
					race_session = get_race_session(selected_year, event_name, 'R')
					print("Data loaded successfully")
				except Exception as e:
					print(f"Error loading race data: {str(e)}"); print("Using qualifying session data instead."); race_session = session
			show_race_results(race_session)
		elif choice == '2':
			if race_session is None:
				print("\nLoading race session data for driver comparison")
				try:
					race_session = get_race_session(selected_year, event_name, 'R')
					print("Race data loaded successfully")
				except Exception as e:
					print(f"Error loading race data: {str(e)}"); print("Using qualifying session data instead."); race_session = session
			driver1 = input("Enter first driver's initials (e.g., VER): ").strip().upper()
			driver2 = input("Enter second driver's initials (e.g., HAM): ").strip().upper()
			compare_drivers(race_session, driver1, driver2)
		elif choice == '3':
			if race_session is None:
				print("\nLoading race session data for tire strategies")
				try:
					race_session = get_race_session(selected_year, event_name, 'R')
					print("Race data loaded successfully")
				except Exception as e:
					print(f"Error loading race data: {str(e)}"); print("Using qualifying session data instead."); race_session = session
			show_tire_strategies(race_session)
		elif choice == '4':
			show_session_info(session)
		elif choice == '5':
			driver = input("Enter driver's initials (e.g., VER): ").strip().upper()
			print("\nNote: Q3 prediction uses 2024 data for model training regardless of selected year.")
			predict_q3_qualification(session, driver)
		elif choice == '6':
			driver = input("Enter driver's initials (e.g., VER): ").strip().upper()
			team = input("Enter team name (e.g., RED BULL RACING): ").strip().upper()
			print("\nNote: Model uses 2022-2023 data for training.")
			predict_2025_qualification(driver, team, f"{event_name} 2025")
		elif choice == '7':
			print("\nTraining models on 2022-2023 data and testing on 2024...")
			print("This will take a few minutes but only needs to be done once.")
			confirm = input("Continue? (y/n): ").strip().lower()
			if confirm == 'y':
				try:
					train_models_once()
					print("\nModels trained and saved successfully!")
					print("You can now use options 5 and 6 for predictions.")
				except Exception as e:
					print(f"\nTraining failed: {str(e)}")
			else:
				print("Training cancelled.")
		elif choice == '8':
			print("Exiting"); break
		else:
			print("Invalid option. Please try again.")

if __name__ == "__main__":
	analyze_current_season()


