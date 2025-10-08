#!/usr/bin/env python3
"""
Test runner script for PCB Defect Detection API
"""

import subprocess
import sys
import os

def run_tests(test_type="all"):
    """Run tests based on type"""
    
    if test_type == "unit":
        cmd = ["python", "-m", "pytest", "tests/test_config.py", "tests/test_image_processor.py", "tests/test_response_formatter.py", "-v"]
    elif test_type == "integration":
        cmd = ["python", "-m", "pytest", "tests/test_integration.py", "-v"]
    elif test_type == "api":
        cmd = ["python", "-m", "pytest", "tests/test_api.py", "-v"]
    elif test_type == "all":
        cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    print(f"Running {test_type} tests...")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
    else:
        test_type = "all"
    
    success = run_tests(test_type)
    
    if success:
        print(f"\n✅ {test_type.title()} tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {test_type.title()} tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
