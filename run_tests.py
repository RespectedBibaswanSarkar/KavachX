#!/usr/bin/env python3
"""
Comprehensive test script for KavachX system.
Tests all 5 layers + API integration.
"""

import sys
import os

# Set working directory
os.chdir(r'd:\MASTER_BIBASWAN_SARKAR\IIIT MANIPUR\PSB HACKATHON\KavachX')
sys.path.insert(0, '.')

def run_tests():
    """Run all system tests."""
    
    print("\n" + "="*80)
    print("KAVACHX SYSTEM INTEGRATION TEST")
    print("="*80)
    
    # Test Layer 1: Static Rules
    print("\n[TEST 1/5] Layer 1 - Static Rule Engine")
    print("-" * 80)
    try:
        from rules import demo_static_rules
        demo_static_rules()
        print("✓ Layer 1 tests passed")
    except Exception as e:
        print(f"✗ Layer 1 test failed: {e}")
        return False
    
    # Test Layer 2: Features
    print("\n[TEST 2/5] Layer 2 - Feature Extraction")
    print("-" * 80)
    try:
        from features import demo_feature_extraction
        demo_feature_extraction()
        print("✓ Layer 2 tests passed")
    except Exception as e:
        print(f"✗ Layer 2 test failed: {e}")
        return False
    
    # Test Layer 3: RL Agent
    print("\n[TEST 3/5] Layer 3 - RL Agent (Dueling Double DQN)")
    print("-" * 80)
    try:
        from rl_agent import demo_rl_agent
        demo_rl_agent()
        print("✓ Layer 3 tests passed")
    except Exception as e:
        print(f"✗ Layer 3 test failed: {e}")
        return False
    
    # Test Layer 4: Classical ML
    print("\n[TEST 4/5] Layer 4 - Classical ML Pipeline")
    print("-" * 80)
    try:
        from classical_ml import demo_classical_ml
        demo_classical_ml()
        print("✓ Layer 4 tests passed")
    except Exception as e:
        print(f"✗ Layer 4 test failed: {e}")
        return False
    
    # Test Layer 5: Fusion
    print("\n[TEST 5/5] Layer 5 - Ensemble Fusion")
    print("-" * 80)
    try:
        from fusion import demo_ensemble_fusion
        demo_ensemble_fusion()
        print("✓ Layer 5 tests passed")
    except Exception as e:
        print(f"✗ Layer 5 test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_tests()
    
    print("\n" + "="*80)
    if success:
        print("✓ ALL SYSTEM TESTS PASSED")
        print("="*80)
        print("\nNext step: Run API server demo with:")
        print("  python api_server.py")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        sys.exit(1)
