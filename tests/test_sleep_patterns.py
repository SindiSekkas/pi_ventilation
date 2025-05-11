#!/usr/bin/env python3
"""Test script for sleep patterns system."""
import os
import sys
import logging
import json
import shutil
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("sleep_patterns_test")

def setup_test_environment():
    """Create test directory and initialize test data."""
    test_data_dir = "test_data/sleep_patterns_test"
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
    os.makedirs(test_data_dir)
    return test_data_dir

class MockDataManager:
    """Mock data manager for sleep pattern tests."""
    def __init__(self):
        self.latest_data = {
            "scd41": {
                "co2": 600,
                "temperature": 22.0,
                "humidity": 50.0
            },
            "room": {
                "occupants": 1,
                "ventilated": False
            }
        }
    
    def update_co2(self, co2_value):
        """Helper to update CO2 value for testing."""
        self.latest_data["scd41"]["co2"] = co2_value

class MockController:
    """Mock controller for sleep pattern tests."""
    def __init__(self):
        self.night_mode_enabled = True
        self.night_mode_start_hour = 23
        self.night_mode_end_hour = 7
    
    def get_status(self):
        """Return mock status."""
        return {
            "night_mode": {
                "enabled": self.night_mode_enabled,
                "start_hour": self.night_mode_start_hour,
                "end_hour": self.night_mode_end_hour,
                "currently_active": False
            }
        }
    
    def set_night_mode(self, enabled=None, start_hour=None, end_hour=None):
        """Mock set night mode."""
        if enabled is not None:
            self.night_mode_enabled = enabled
        if start_hour is not None:
            self.night_mode_start_hour = start_hour
        if end_hour is not None:
            self.night_mode_end_hour = end_hour
        return True

def test_pattern_initialization():
    """Test sleep pattern initialization and basic data structure."""
    logger.info("Testing pattern initialization...")
    
    test_dir = setup_test_environment()
    
    try:
        from predictive.adaptive_sleep_analyzer import AdaptiveSleepAnalyzer
        
        mock_data_manager = MockDataManager()
        mock_controller = MockController()
        
        # Initialize analyzer
        analyzer = AdaptiveSleepAnalyzer(mock_data_manager, mock_controller)
        analyzer.data_dir = test_dir
        analyzer.sleep_patterns_file = os.path.join(test_dir, "adaptive_sleep_patterns.json")
        
        # Check that patterns were initialized
        assert "weekday_patterns" in analyzer.sleep_patterns
        assert "detected_events" in analyzer.sleep_patterns
        assert "daily_patterns" in analyzer.sleep_patterns
        
        # Check all weekdays are present
        for i in range(7):
            assert str(i) in analyzer.sleep_patterns["weekday_patterns"]
            weekday_pattern = analyzer.sleep_patterns["weekday_patterns"][str(i)]
            assert weekday_pattern["sleep"] is None
            assert weekday_pattern["wake"] is None
            assert weekday_pattern["confidence"] == 0
            assert weekday_pattern["detections"] == 0
        
        logger.info("✅ Pattern initialization successful")
        
        # Test save and load
        analyzer.save_patterns()
        
        # Create new analyzer and load patterns
        new_analyzer = AdaptiveSleepAnalyzer(mock_data_manager, mock_controller)
        new_analyzer.data_dir = test_dir
        new_analyzer.sleep_patterns_file = os.path.join(test_dir, "adaptive_sleep_patterns.json")
        
        # Verify loaded patterns
        assert "weekday_patterns" in new_analyzer.sleep_patterns
        assert len(new_analyzer.sleep_patterns["weekday_patterns"]) == 7
        logger.info("✅ Pattern save/load working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"Pattern initialization test failed: {e}", exc_info=True)
        return False

def test_sleep_event_logging():
    """Test sleep event detection and logging."""
    logger.info("Testing sleep event logging...")
    
    test_dir = setup_test_environment()
    
    try:
        from predictive.adaptive_sleep_analyzer import AdaptiveSleepAnalyzer
        
        mock_data_manager = MockDataManager()
        mock_controller = MockController()
        
        analyzer = AdaptiveSleepAnalyzer(mock_data_manager, mock_controller)
        analyzer.data_dir = test_dir
        analyzer.sleep_patterns_file = os.path.join(test_dir, "adaptive_sleep_patterns.json")
        
        # Manually log a sleep event
        test_time = datetime.now().replace(hour=23, minute=15)
        event_details = {
            "rate_before": 3.5,
            "rate_after": 1.2,
            "var_before": 0.8,
            "var_after": 0.3,
            "confidence": 0.85
        }
        
        analyzer._log_sleep_event("sleep_start", test_time, event_details)
        
        # Check that event was logged
        assert len(analyzer.sleep_patterns["detected_events"]) == 1
        event = analyzer.sleep_patterns["detected_events"][0]
        assert event["type"] == "sleep_start"
        assert event["weekday"] == test_time.weekday()
        assert event["details"]["confidence"] == 0.85
        
        # Check weekday pattern was updated
        weekday_key = str(test_time.weekday())
        pattern = analyzer.sleep_patterns["weekday_patterns"][weekday_key]
        assert pattern["sleep"] == "23:15"
        assert pattern["detections"] == 1
        assert pattern["confidence"] == 0.85
        
        logger.info("✅ Sleep event logging working correctly")
        
        # Log a wake event
        wake_time = test_time.replace(hour=7, minute=15)  # Keep on same day
        wake_details = {
            "rate_before": 0.8,
            "rate_after": 3.2,
            "var_before": 0.3,
            "var_after": 0.9,
            "confidence": 0.90
        }
        
        analyzer._log_sleep_event("wake_up", wake_time, wake_details)
        
        # Check wake event was logged
        assert len(analyzer.sleep_patterns["detected_events"]) == 2
        wake_event = analyzer.sleep_patterns["detected_events"][1]
        assert wake_event["type"] == "wake_up"
        assert wake_event["details"]["confidence"] == 0.90
        
        # Check wake pattern was updated
        pattern = analyzer.sleep_patterns["weekday_patterns"][weekday_key]
        assert pattern["wake"] == "07:15"
        assert pattern["detections"] == 2
        assert pattern["confidence"] == 0.90  # Should be max of sleep/wake confidence
        
        logger.info("✅ Wake event logging working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"Sleep event logging test failed: {e}", exc_info=True)
        return False

def test_confidence_calculations():
    """Test confidence calculation for sleep/wake predictions."""
    logger.info("Testing confidence calculations...")
    
    test_dir = setup_test_environment()
    
    try:
        from predictive.adaptive_sleep_analyzer import AdaptiveSleepAnalyzer
        
        mock_data_manager = MockDataManager()
        mock_controller = MockController()
        
        analyzer = AdaptiveSleepAnalyzer(mock_data_manager, mock_controller)
        analyzer.data_dir = test_dir
        analyzer.sleep_patterns_file = os.path.join(test_dir, "adaptive_sleep_patterns.json")
        
        # Create sleep patterns for Monday with multiple detections
        monday_key = "0"
        analyzer.sleep_patterns["weekday_patterns"][monday_key] = {
            "sleep": "23:00",
            "wake": "07:00",
            "confidence": 0.85,
            "detections": 10  # High number of detections
        }
        
        # Create some daily patterns for variance calculation
        base_date = datetime.now().date()
        for i in range(5):
            date_str = (base_date - timedelta(days=i*7)).isoformat()
            analyzer.sleep_patterns["daily_patterns"][date_str] = {
                "sleep": "23:00",  # Consistent time
                "wake": "07:00",   # Consistent time
                "weekday": 0,
                "sleep_confidence": 0.8,
                "wake_confidence": 0.8
            }
        
        # Add recent events
        for i in range(3):
            event_time = datetime.now() - timedelta(days=i*7)
            analyzer.sleep_patterns["detected_events"].append({
                "type": "sleep_start",
                "timestamp": event_time.isoformat(),
                "weekday": 0,
                "details": {"confidence": 0.8}
            })
        
        # Test confidence calculation for Monday
        sleep_time, sleep_confidence = analyzer.get_predicted_sleep_time_for_day(0)
        wake_time, wake_confidence = analyzer.get_predicted_wake_time_for_day(0)
        
        assert sleep_time is not None
        assert wake_time is not None
        assert 0.7 <= sleep_confidence <= 0.95  # Should be high confidence
        assert 0.7 <= wake_confidence <= 0.95   # Should be high confidence
        
        logger.info(f"✅ Sleep confidence: {sleep_confidence:.2f}, Wake confidence: {wake_confidence:.2f}")
        
        # Test with no data (should return low confidence)
        sleep_time_none, sleep_confidence_none = analyzer.get_predicted_sleep_time_for_day(6)  # Sunday
        wake_time_none, wake_confidence_none = analyzer.get_predicted_wake_time_for_day(6)    # Sunday
        
        assert sleep_time_none is None
        assert wake_time_none is None
        assert sleep_confidence_none == 0.0
        assert wake_confidence_none == 0.0
        
        logger.info("✅ No data returns 0 confidence correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"Confidence calculation test failed: {e}", exc_info=True)
        return False

def test_sleep_pattern_summary():
    """Test getting sleep pattern summary with confidence values."""
    logger.info("Testing sleep pattern summary...")
    
    test_dir = setup_test_environment()
    
    try:
        from predictive.adaptive_sleep_analyzer import AdaptiveSleepAnalyzer
        
        mock_data_manager = MockDataManager()
        mock_controller = MockController()
        
        analyzer = AdaptiveSleepAnalyzer(mock_data_manager, mock_controller)
        analyzer.data_dir = test_dir
        analyzer.sleep_patterns_file = os.path.join(test_dir, "adaptive_sleep_patterns.json")
        
        # Set up some patterns
        analyzer.sleep_patterns["weekday_patterns"]["0"] = {
            "sleep": "23:00",
            "wake": "07:00",
            "confidence": 0.85,
            "detections": 10
        }
        
        analyzer.sleep_patterns["weekday_patterns"]["1"] = {
            "sleep": "23:30",
            "wake": "07:30",
            "confidence": 0.90,
            "detections": 8
        }
        
        # Add some recent events
        recent_time = datetime.now() - timedelta(hours=2)
        analyzer.sleep_patterns["detected_events"] = [
            {
                "type": "sleep_start",
                "timestamp": recent_time.isoformat(),
                "weekday": 0,
                "details": {"confidence": 0.85}
            }
        ]
        
        # Add adjustment history
        analyzer.sleep_patterns["night_mode_adjustments"] = [
            {
                "timestamp": datetime.now().isoformat(),
                "type": "start_time",
                "from": 23,
                "to": 22,
                "detected_time": "22:45",
                "confidence": 0.85
            }
        ]
        
        # Get summary
        summary = analyzer.get_sleep_pattern_summary()
        
        # Verify summary structure
        assert "weekday_patterns" in summary
        assert "recent_events" in summary
        assert "recent_adjustments" in summary
        assert "current_night_mode" in summary
        
        # Check weekday patterns with confidence
        assert "Monday" in summary["weekday_patterns"]
        assert "Tuesday" in summary["weekday_patterns"]
        
        monday_pattern = summary["weekday_patterns"]["Monday"]
        assert monday_pattern["sleep"] == "23:00"
        assert monday_pattern["wake"] == "07:00"
        assert "sleep_confidence" in monday_pattern
        assert "wake_confidence" in monday_pattern
        assert 0.7 <= monday_pattern["sleep_confidence"] <= 0.95
        assert 0.7 <= monday_pattern["wake_confidence"] <= 0.95
        
        # Check recent events
        assert len(summary["recent_events"]) == 1
        assert summary["recent_events"][0]["type"] == "sleep_start"
        assert summary["recent_events"][0]["confidence"] == "0.85"
        
        # Check adjustments
        assert len(summary["recent_adjustments"]) == 1
        assert summary["recent_adjustments"][0]["type"] == "start_time"
        assert summary["recent_adjustments"][0]["from"] == "23:00"
        assert summary["recent_adjustments"][0]["to"] == "22:00"
        
        logger.info("✅ Sleep pattern summary working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"Sleep pattern summary test failed: {e}", exc_info=True)
        return False

def test_night_mode_adjustments():
    """Test night mode time adjustments based on detected patterns."""
    logger.info("Testing night mode adjustments...")
    
    test_dir = setup_test_environment()
    
    try:
        from predictive.adaptive_sleep_analyzer import AdaptiveSleepAnalyzer
        
        mock_data_manager = MockDataManager()
        mock_controller = MockController()
        
        analyzer = AdaptiveSleepAnalyzer(mock_data_manager, mock_controller)
        analyzer.data_dir = test_dir
        analyzer.sleep_patterns_file = os.path.join(test_dir, "adaptive_sleep_patterns.json")
        
        # Set required detections for adjustment
        analyzer.required_detections = 1  # Lower threshold for testing
        
        # Set a weekday pattern with enough detections
        weekday_key = "0"  # Monday
        analyzer.sleep_patterns["weekday_patterns"][weekday_key] = {
            "sleep": "22:30",  # Earlier than current 23:00
            "wake": "06:45",   # Earlier than current 07:00
            "confidence": 0.85,
            "detections": 5
        }
        
        # Trigger adjustment for start time
        detected_sleep_time = datetime.now().replace(hour=22, minute=30)
        success = analyzer._adjust_night_start_time(detected_sleep_time, 0.85)
        assert success
        
        # Check that controller was updated
        status = mock_controller.get_status()
        assert status["night_mode"]["start_hour"] == 22  # Should be adjusted to 22
        
        # Check adjustment was logged
        assert len(analyzer.sleep_patterns["night_mode_adjustments"]) == 1
        adjustment = analyzer.sleep_patterns["night_mode_adjustments"][0]
        assert adjustment["type"] == "start_time"
        assert adjustment["from"] == 23
        assert adjustment["to"] == 22
        
        logger.info("✅ Night mode start time adjustment working")
        
        # Trigger adjustment for end time
        detected_wake_time = datetime.now().replace(hour=6, minute=45)
        success = analyzer._adjust_night_end_time(detected_wake_time, 0.85)
        assert success
        
        # Check that controller was updated
        status = mock_controller.get_status()
        assert status["night_mode"]["end_hour"] == 6  # Should be adjusted to 6
        
        # Check adjustment was logged
        assert len(analyzer.sleep_patterns["night_mode_adjustments"]) == 2
        adjustment = analyzer.sleep_patterns["night_mode_adjustments"][1]
        assert adjustment["type"] == "end_time"
        assert adjustment["from"] == 7
        assert adjustment["to"] == 6
        
        logger.info("✅ Night mode end time adjustment working")
        
        return True
        
    except Exception as e:
        logger.error(f"Night mode adjustment test failed: {e}", exc_info=True)
        return False

def run_all_tests():
    """Run all sleep pattern tests."""
    tests = [
        ("Pattern Initialization", test_pattern_initialization),
        ("Sleep Event Logging", test_sleep_event_logging),
        ("Confidence Calculations", test_confidence_calculations),
        ("Sleep Pattern Summary", test_sleep_pattern_summary),
        ("Night Mode Adjustments", test_night_mode_adjustments)
    ]
    
    print("\n===== SLEEP PATTERNS TEST =====\n")
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            print(f"✅ {test_name} PASSED\n")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED\n")
            failed += 1
    
    print("===== TEST RESULTS =====")
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    # Clean up test directory
    test_dir = "test_data/sleep_patterns_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    return failed == 0

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)