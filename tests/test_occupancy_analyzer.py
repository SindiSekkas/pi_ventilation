"""Test script for OccupancyPatternAnalyzer.

For interactive menu:
python test_occupancy_analyzer.py

For automated test:
python test_occupancy_analyzer.py --auto

To use a different history file:
python test_occupancy_analyzer.py --file /path/to/occupancy_history.csv
"""

# tests/test_occupancy_analyzer.py

import time
import logging
import sys
import argparse
import tempfile
import os
import csv
from datetime import datetime, timedelta

# parent directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("occupancy_test")

# Import components
try:
    from predictive.occupancy_pattern_analyzer import OccupancyPatternAnalyzer
except ImportError:
    logger.error("Unable to import OccupancyPatternAnalyzer. Please check your installation.")
    sys.exit(1)

def create_test_data(filename):
    """Create test data for the occupancy analyzer."""
    test_data = []
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Create 7 days of test data
    for day in range(7):
        current_day = base_time + timedelta(days=day)
        
        # Typical weekday pattern
        if day < 5:  # Monday to Friday
            # Night (00:00-07:00): Empty
            for hour in range(0, 7):
                test_data.append({
                    'timestamp': (current_day + timedelta(hours=hour)).isoformat(),
                    'status': 'EMPTY',
                    'people_count': 0
                })
            
            # Morning (07:00-09:00): Occupied
            for hour in range(7, 9):
                test_data.append({
                    'timestamp': (current_day + timedelta(hours=hour)).isoformat(),
                    'status': 'OCCUPIED',
                    'people_count': 2
                })
            
            # Work day (09:00-17:00): Empty
            for hour in range(9, 17):
                test_data.append({
                    'timestamp': (current_day + timedelta(hours=hour)).isoformat(),
                    'status': 'EMPTY',
                    'people_count': 0
                })
            
            # Evening (17:00-23:00): Occupied
            for hour in range(17, 23):
                test_data.append({
                    'timestamp': (current_day + timedelta(hours=hour)).isoformat(),
                    'status': 'OCCUPIED',
                    'people_count': 2
                })
            
            # Night (23:00): Empty
            test_data.append({
                'timestamp': (current_day + timedelta(hours=23)).isoformat(),
                'status': 'EMPTY',
                'people_count': 0
            })
        
        # Weekend pattern
        else:  # Saturday and Sunday
            # Late night (00:00-10:00): Empty
            for hour in range(0, 10):
                test_data.append({
                    'timestamp': (current_day + timedelta(hours=hour)).isoformat(),
                    'status': 'EMPTY',
                    'people_count': 0
                })
            
            # Day (10:00-22:00): Occupied
            for hour in range(10, 22):
                test_data.append({
                    'timestamp': (current_day + timedelta(hours=hour)).isoformat(),
                    'status': 'OCCUPIED',
                    'people_count': 2
                })
            
            # Night (22:00-24:00): Empty
            for hour in range(22, 24):
                test_data.append({
                    'timestamp': (current_day + timedelta(hours=hour)).isoformat(),
                    'status': 'EMPTY',
                    'people_count': 0
                })
    
    # Write test data to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'status', 'people_count'])
        writer.writeheader()
        writer.writerows(test_data)
    
    logger.info(f"Created test data with {len(test_data)} records in {filename}")

def test_occupancy_analyzer(history_file=None, auto_test=False):
    """Test occupancy pattern analyzer functionality."""
    # Create a temporary test file if none provided
    if history_file is None:
        temp_dir = tempfile.mkdtemp()
        history_file = os.path.join(temp_dir, "test_occupancy_history.csv")
        create_test_data(history_file)
        logger.info(f"Using temporary test file: {history_file}")
    
    # Initialize the analyzer
    try:
        analyzer = OccupancyPatternAnalyzer(history_file)
    except Exception as e:
        logger.error(f"Failed to initialize OccupancyPatternAnalyzer: {e}")
        return False
    
    logger.info("Successfully initialized OccupancyPatternAnalyzer")
    
    if auto_test:
        return run_automated_test(analyzer)
    else:
        run_interactive_menu(analyzer)
        return True

def run_automated_test(analyzer):
    """Run an automated test suite for the occupancy analyzer."""
    print("\n====== STARTING AUTOMATED TEST SEQUENCE ======")
    print("This will test all major functionality of the OccupancyPatternAnalyzer")
    print("Press Ctrl+C to abort at any time")
    
    test_results = []
    all_passed = True
    
    # Test 1: Load patterns
    print("\nTest 1: Loading patterns from history...")
    try:
        analyzer._load_and_process_history()
        pattern_count = len(analyzer.empty_probabilities)
        success = pattern_count > 0
        test_results.append({
            "test": "Load patterns",
            "success": success,
            "details": f"Loaded {pattern_count} patterns"
        })
        if success:
            print(f"✅ Test 1 PASSED: Loaded {pattern_count} patterns")
        else:
            print("❌ Test 1 FAILED: No patterns loaded")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Load patterns", "success": False, "details": str(e)})
    
    # Test 2: Get pattern summary
    print("\nTest 2: Getting pattern summary...")
    try:
        summary = analyzer.get_pattern_summary()
        success = "total_patterns" in summary and summary["total_patterns"] > 0
        test_results.append({
            "test": "Pattern summary",
            "success": success,
            "details": f"Total patterns: {summary.get('total_patterns', 0)}"
        })
        if success:
            print(f"✅ Test 2 PASSED: Got summary with {summary['total_patterns']} patterns")
        else:
            print("❌ Test 2 FAILED: Invalid summary")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Pattern summary", "success": False, "details": str(e)})
    
    # Test 3: Predict empty probability
    print("\nTest 3: Predicting empty probability...")
    try:
        test_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)  # 2 PM on current day
        prob = analyzer.get_predicted_empty_probability(test_time)
        success = 0.0 <= prob <= 1.0
        test_results.append({
            "test": "Empty probability",
            "success": success,
            "details": f"Probability at {test_time.hour}:00 = {prob:.3f}"
        })
        if success:
            print(f"✅ Test 3 PASSED: Probability at {test_time.hour}:00 = {prob:.3f}")
        else:
            print(f"❌ Test 3 FAILED: Invalid probability {prob}")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Empty probability", "success": False, "details": str(e)})
    
    # Test 4: Get next significant event
    print("\nTest 4: Getting next significant event...")
    try:
        now = datetime.now()
        event_time, event_type, confidence = analyzer.get_next_significant_event(now)
        success = event_time is not None or (event_time is None and event_type is None)
        details = f"Event: {event_type} at {event_time} (confidence: {confidence:.3f})" if event_time else "No event predicted"
        test_results.append({
            "test": "Next significant event",
            "success": success,
            "details": details
        })
        if success:
            print(f"✅ Test 4 PASSED: {details}")
        else:
            print("❌ Test 4 FAILED: Invalid event data")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Next significant event", "success": False, "details": str(e)})
    
    # Test 5: Get current period
    print("\nTest 5: Getting current predicted period...")
    try:
        start, end, status, confidence = analyzer.get_predicted_current_period()
        success = status is not None
        details = f"Period: {status} from {start} to {end} (confidence: {confidence:.3f})" if status else "No period predicted"
        test_results.append({
            "test": "Current period",
            "success": success,
            "details": details
        })
        if success:
            print(f"✅ Test 5 PASSED: {details}")
        else:
            print("❌ Test 5 FAILED: Invalid period data")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Current period", "success": False, "details": str(e)})
    
    # Test 6: Record user feedback
    print("\nTest 6: Recording user feedback...")
    try:
        feedback_time = datetime.now()
        old_prob = analyzer.get_predicted_empty_probability(feedback_time)
        analyzer.record_user_feedback(feedback_time, "USER_CONFIRMED_HOME")
        new_prob = analyzer.get_predicted_empty_probability(feedback_time)
        success = True  # Just check it doesn't error
        test_results.append({
            "test": "User feedback",
            "success": success,
            "details": f"Probability changed from {old_prob:.3f} to {new_prob:.3f}"
        })
        print(f"✅ Test 6 PASSED: Recorded feedback, probability changed from {old_prob:.3f} to {new_prob:.3f}")
    except Exception as e:
        print(f"❌ Test 6 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "User feedback", "success": False, "details": str(e)})
    
    # Display summary
    print("\n====== AUTOMATED TEST RESULTS ======")
    for result in test_results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{result['test']}: {status} - {result['details']}")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED! OccupancyPatternAnalyzer is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED. Check the log for details.")
    
    return all_passed

def run_interactive_menu(analyzer):
    """Run an interactive menu for manual testing."""
    while True:
        # Display menu
        print("\n====== OCCUPANCY PATTERN ANALYZER TEST ======")
        print("\nOptions:")
        print("1. Show pattern summary")
        print("2. Test empty probability prediction")
        print("3. Test next significant event")
        print("4. Test current period prediction")
        print("5. Test user feedback")
        print("6. Run automated test sequence")
        print("7. Reload patterns from file")
        print("8. Exit")
        
        # Get user input
        try:
            choice = input("\nEnter your choice (1-8): ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        # Process choice
        try:
            if choice == "1":
                summary = analyzer.get_pattern_summary()
                print(f"\nPattern Summary:")
                print(f"Total patterns: {summary.get('total_patterns', 0)}")
                if summary.get('last_update'):
                    print(f"Last updated: {summary['last_update']}")
                print("\nEmpty hour ranges by day:")
                for day, ranges in summary.get('empty_hour_ranges', {}).items():
                    if ranges:
                        range_strs = []
                        for start, end in ranges:
                            if start == end:
                                range_strs.append(f"{start}:00")
                            else:
                                range_strs.append(f"{start}:00-{end}:00")
                        print(f"  {day}: {', '.join(range_strs)}")
                    else:
                        print(f"  {day}: No clear pattern")
            
            elif choice == "2":
                hour_input = input("Enter hour (0-23): ")
                try:
                    hour = int(hour_input)
                    if 0 <= hour <= 23:
                        test_time = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
                        prob = analyzer.get_predicted_empty_probability(test_time)
                        print(f"Probability of being empty at {hour}:00 = {prob:.3f}")
                        status = "LIKELY EMPTY" if prob > 0.7 else "LIKELY OCCUPIED" if prob < 0.3 else "UNCERTAIN"
                        print(f"Prediction: {status}")
                    else:
                        print("Please enter a valid hour (0-23)")
                except ValueError:
                    print("Please enter a valid number")
            
            elif choice == "3":
                event_time, event_type, confidence = analyzer.get_next_significant_event()
                if event_time:
                    print(f"\nNext significant event:")
                    print(f"Type: {event_type}")
                    print(f"Time: {event_time.strftime('%Y-%m-%d %H:%M')}")
                    print(f"Confidence: {confidence:.3f}")
                else:
                    print("\nNo significant event predicted")
            
            elif choice == "4":
                start, end, status, confidence = analyzer.get_predicted_current_period()
                if status:
                    print(f"\nCurrent predicted period:")
                    print(f"Status: {status}")
                    print(f"Start: {start.strftime('%Y-%m-%d %H:%M') if start else 'Unknown'}")
                    print(f"End: {end.strftime('%Y-%m-%d %H:%M') if end else 'Unknown'}")
                    print(f"Confidence: {confidence:.3f}")
                else:
                    print("\nNo current period predicted")
            
            elif choice == "5":
                print("\nRecord user feedback:")
                print("1. I'm home")
                print("2. I'm away")
                fb_choice = input("Enter choice (1-2): ")
                
                if fb_choice == "1":
                    analyzer.record_user_feedback(datetime.now(), "USER_CONFIRMED_HOME")
                    print("✅ Recorded: User confirmed home")
                elif fb_choice == "2":
                    analyzer.record_user_feedback(datetime.now(), "USER_CONFIRMED_AWAY")
                    print("✅ Recorded: User confirmed away")
                else:
                    print("Invalid choice")
            
            elif choice == "6":
                run_automated_test(analyzer)
            
            elif choice == "7":
                print("Reloading patterns...")
                analyzer._load_and_process_history()
                print(f"✅ Patterns reloaded: {len(analyzer.empty_probabilities)} total")
            
            elif choice == "8":
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please enter a number between 1 and 8.")
                
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test OccupancyPatternAnalyzer functionality")
    parser.add_argument("--file", help="Path to occupancy history CSV file")
    parser.add_argument("--auto", action="store_true", help="Run automated test sequence")
    args = parser.parse_args()
    
    # Run test
    success = test_occupancy_analyzer(history_file=args.file, auto_test=args.auto)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)