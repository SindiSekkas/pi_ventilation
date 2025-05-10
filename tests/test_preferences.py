#!/usr/bin/env python3
"""Test script for preferences system."""
import os
import sys
import logging
import json
import shutil
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("preferences_test")

def setup_test_environment():
    """Create test directory and initialize test data."""
    test_data_dir = "test_data/preferences_test"
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
    os.makedirs(test_data_dir)
    return test_data_dir

def test_basic_preference_operations():
    """Test basic preference creation, loading, and saving."""
    logger.info("Testing basic preference operations...")
    
    test_dir = setup_test_environment()
    
    try:
        from preferences.preference_manager import PreferenceManager
        from preferences.models import UserPreference
        
        # Initialize PreferenceManager
        manager = PreferenceManager(data_dir=test_dir)
        
        # Test creating a new user preference
        user_id = 12345
        user_pref = manager.get_user_preference(user_id, "Test User")
        
        # Check default values
        assert user_pref.temp_min == 20.0
        assert user_pref.temp_max == 24.0
        assert user_pref.co2_threshold == 1000
        assert user_pref.username == "Test User"
        logger.info("✅ Default preference values correct")
        
        # Test updating preferences
        success = manager.set_user_preference(
            user_id,
            temp_min=21.0,
            temp_max=23.0,
            co2_threshold=850
        )
        assert success
        
        # Verify updated values
        updated_pref = manager.get_user_preference(user_id)
        assert updated_pref.temp_min == 21.0
        assert updated_pref.temp_max == 23.0
        assert updated_pref.co2_threshold == 850
        logger.info("✅ Preference updates working correctly")
        
        # Test persistence (create new manager instance)
        new_manager = PreferenceManager(data_dir=test_dir)
        loaded_pref = new_manager.get_user_preference(user_id)
        assert loaded_pref.temp_min == 21.0
        assert loaded_pref.temp_max == 23.0
        assert loaded_pref.co2_threshold == 850
        logger.info("✅ Preference persistence working")
        
        return True
        
    except Exception as e:
        logger.error(f"Basic preference test failed: {e}", exc_info=True)
        return False

def test_feedback_system():
    """Test user feedback recording and preference adjustment."""
    logger.info("Testing feedback system...")
    
    test_dir = setup_test_environment()
    
    try:
        from preferences.preference_manager import PreferenceManager
        
        manager = PreferenceManager(data_dir=test_dir)
        user_id = 12345
        
        # Get initial preferences
        initial_pref = manager.get_user_preference(user_id)
        initial_temp_max = initial_pref.temp_max
        initial_co2_threshold = initial_pref.co2_threshold
        
        # Simulate sensor data
        sensor_data = {
            "scd41": {
                "temperature": 25.0,
                "co2": 1200,
                "humidity": 55.0
            },
            "room": {
                "occupants": 2
            }
        }
        
        # Test "too hot" feedback
        success = manager.update_preference_from_feedback(
            user_id,
            "too_hot",
            sensor_data
        )
        assert success
        
        # Check that temp_max was reduced
        updated_pref = manager.get_user_preference(user_id)
        assert updated_pref.temp_max < initial_temp_max
        logger.info("✅ Too hot feedback correctly reduced temp_max")
        
        # Test "stuffy" feedback
        success = manager.update_preference_from_feedback(
            user_id,
            "stuffy",
            sensor_data
        )
        assert success
        
        # Check that CO2 threshold was reduced
        updated_pref = manager.get_user_preference(user_id)
        assert updated_pref.co2_threshold < initial_co2_threshold
        logger.info("✅ Stuffy feedback correctly reduced CO2 threshold")
        
        # Test feedback history
        feedback_history = manager.get_user_feedback_history(user_id)
        assert len(feedback_history) == 2
        assert feedback_history[0].feedback_type == "too_hot"
        assert feedback_history[1].feedback_type == "stuffy"
        logger.info("✅ Feedback history recording working")
        
        return True
        
    except Exception as e:
        logger.error(f"Feedback system test failed: {e}", exc_info=True)
        return False

def test_compromise_preferences():
    """Test calculating compromise preferences for multiple users."""
    logger.info("Testing compromise preferences...")
    
    test_dir = setup_test_environment()
    
    try:
        from preferences.preference_manager import PreferenceManager
        
        manager = PreferenceManager(data_dir=test_dir)
        
        # Create preferences for three users with different settings
        users = [
            {
                "id": 1,
                "name": "User1",
                "prefs": {
                    "temp_min": 19.0,
                    "temp_max": 22.0,
                    "co2_threshold": 900,
                    "humidity_min": 30.0,
                    "humidity_max": 50.0
                }
            },
            {
                "id": 2,
                "name": "User2",
                "prefs": {
                    "temp_min": 21.0,
                    "temp_max": 24.0,
                    "co2_threshold": 1100,
                    "humidity_min": 40.0,
                    "humidity_max": 65.0
                }
            },
            {
                "id": 3,
                "name": "User3",
                "prefs": {
                    "temp_min": 20.0,
                    "temp_max": 23.0,
                    "co2_threshold": 1000,
                    "humidity_min": 35.0,
                    "humidity_max": 60.0
                }
            }
        ]
        
        # Set preferences for all users
        for user in users:
            manager.set_user_preference(user["id"], **user["prefs"])
        
        # Calculate compromise preferences
        user_ids = [user["id"] for user in users]
        compromise = manager.calculate_compromise_preference(user_ids)
        
        # Verify compromise values are reasonable
        assert 19.0 <= compromise.temp_min <= 21.0
        assert 22.0 <= compromise.temp_max <= 24.0
        assert 900 <= compromise.co2_threshold <= 1100
        logger.info("✅ Compromise preferences within expected ranges")
        
        # Test effectiveness score
        assert 0.0 <= compromise.effectiveness_score <= 1.0
        logger.info(f"✅ Effectiveness score: {compromise.effectiveness_score:.2f}")
        
        # Test with single user (should be their exact preferences)
        single_compromise = manager.calculate_compromise_preference([1])
        pref1 = manager.get_user_preference(1)
        assert single_compromise.temp_min == pref1.temp_min
        assert single_compromise.temp_max == pref1.temp_max
        assert single_compromise.co2_threshold == pref1.co2_threshold
        logger.info("✅ Single user compromise equals user preferences")
        
        # Test with no users (should return defaults)
        empty_compromise = manager.calculate_compromise_preference([])
        assert empty_compromise.user_count == 0
        assert empty_compromise.effectiveness_score == 1.0
        logger.info("✅ Empty user list returns default preferences")
        
        return True
        
    except Exception as e:
        logger.error(f"Compromise preferences test failed: {e}", exc_info=True)
        return False

def test_sensitivity_settings():
    """Test sensitivity settings and their impact on preferences."""
    logger.info("Testing sensitivity settings...")
    
    test_dir = setup_test_environment()
    
    try:
        from preferences.preference_manager import PreferenceManager
        
        manager = PreferenceManager(data_dir=test_dir)
        user_id = 12345
        
        # Set high temperature sensitivity
        manager.set_user_preference(user_id, sensitivity_temp=1.5)
        
        # Set low CO2 sensitivity
        manager.set_user_preference(user_id, sensitivity_co2=0.5)
        
        user_pref = manager.get_user_preference(user_id)
        assert user_pref.sensitivity_temp == 1.5
        assert user_pref.sensitivity_co2 == 0.5
        logger.info("✅ Sensitivity settings updated correctly")
        
        # Test that sensitivity affects feedback adjustments
        initial_temp_max = user_pref.temp_max
        
        sensor_data = {
            "scd41": {
                "temperature": 25.0,
                "co2": 1200,
                "humidity": 55.0
            }
        }
        
        # Test with high sensitivity - should make bigger adjustment
        manager.update_preference_from_feedback(user_id, "too_hot", sensor_data)
        
        # Get updated user preference
        updated_pref = manager.get_user_preference(user_id)
        temp_change = initial_temp_max - updated_pref.temp_max
        
        # Check that adjustment is larger due to high sensitivity
        assert temp_change >= 0.5  # High sensitivity should result in at least 0.5°C change
        logger.info(f"✅ High sensitivity resulted in {temp_change:.1f}°C adjustment")
        
        return True
        
    except Exception as e:
        logger.error(f"Sensitivity settings test failed: {e}", exc_info=True)
        return False

def run_all_tests():
    """Run all preference tests."""
    tests = [
        ("Basic Preference Operations", test_basic_preference_operations),
        ("Feedback System", test_feedback_system),
        ("Compromise Preferences", test_compromise_preferences),
        ("Sensitivity Settings", test_sensitivity_settings)
    ]
    
    print("\n===== PREFERENCES SYSTEM TEST =====\n")
    
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
    test_dir = "test_data/preferences_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    return failed == 0

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)