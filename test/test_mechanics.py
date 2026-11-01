#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved ExaConstit Testing Framework
Automatically discovers TOML test cases and performs comprehensive validation
"""

import subprocess
import os
import multiprocessing
import numpy as np
import pandas as pd
import unittest
import glob
from pathlib import Path
from sys import platform
import toml
from typing import List, Tuple, Dict, Set
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    passed: bool
    missing_files: List[str]
    file_differences: Dict[str, List[str]]
    error_message: str = ""

# Taken from https://github.com/orgs/community/discussions/49224
# but modified slightly as we don't need as strict of a req as the OP in that thread 
# import requests
# 
def is_on_github_actions() -> bool:
    """Check if running on GitHub Actions CI"""
    if "CI" not in os.environ or not os.environ["CI"] or "GITHUB_RUN_ID" not in os.environ:
        return False
    return True

def extract_basename_from_toml(toml_file: str) -> str:
    """
    Extract basename from TOML file, fallback to filename stem if not specified
    
    Args:
        toml_file: Path to TOML configuration file
        
    Returns:
        basename string for output directory naming
    """
    try:
        with open(toml_file, 'r') as f:
            config = toml.load(f)
        
        # Check if basename is explicitly set in TOML
        if 'basename' in config:
            return config['basename']
        else:
            # Fallback to file stem (filename without extension)
            return Path(toml_file).stem
            
    except Exception as e:
        logger.warning(f"Could not parse {toml_file} for basename: {e}")
        return Path(toml_file).stem

def discover_test_cases(test_dir: str = ".") -> List[Tuple[str, str]]:
    """
    Automatically discover all TOML test cases in the test directory
    
    Args:
        test_dir: Directory to search for TOML files
        
    Returns:
        List of (toml_file, basename) tuples
    """
    toml_files = glob.glob(os.path.join(test_dir, "*.toml"))
    test_cases = []
    
    for toml_file in toml_files:
        basename = extract_basename_from_toml(toml_file)
        test_cases.append((os.path.basename(toml_file), basename))
        logger.info(f"Discovered test case: {toml_file} -> basename: {basename}")
    
    if not test_cases:
        logger.warning(f"No TOML files found in {test_dir}")
    
    return test_cases

def load_data_file(file_path: str) -> np.ndarray:
    """
    Load data file using pandas with automatic header detection
    
    Args:
        file_path: Path to data file
        
    Returns:
        numpy array of numeric data (excluding headers)
    """
    try:
        # Try to read with pandas, handling various formats
        # Use sep='\s+' instead of deprecated delim_whitespace=True
        df = pd.read_csv(file_path, sep=r'\s+', comment='#', 
                        header=0, na_values=['nan', 'NaN', 'inf', '-inf'])
        
        # Convert to numeric, replacing non-numeric with NaN
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        
        # Return as numpy array, dropping any rows with all NaN
        data = df_numeric.dropna(how='all').values
        
        if data.size == 0:
            raise ValueError("No numeric data found in file")
            
        return data
        
    except Exception as e:
        # Fallback: try numpy loadtxt with skip_header
        try:
            return np.loadtxt(file_path, skiprows=1)
        except Exception as e2:
            raise ValueError(f"Could not load {file_path}: {e}, {e2}")

def compare_files(baseline_file: str, result_file: str, rel_tolerance: float = 1e-8, 
                  abs_tolerance: float = 1e-10) -> List[str]:
    """
    Compare two data files with specified relative and absolute tolerances
    
    Args:
        baseline_file: Path to baseline reference file
        result_file: Path to test result file
        rel_tolerance: Relative tolerance for comparison
        abs_tolerance: Absolute tolerance for comparison (for small values)
        
    Returns:
        List of difference descriptions (empty if files match)
    """
    differences = []

    # Skewness and kurtosis divide by powers of the spread, so components with a tiny spread
    # (e.g. the off-diagonal deformation gradient on the first step) magnify solver round-off
    # differences between 1 and 2 ranks
    if any(s in os.path.basename(baseline_file) for s in ("_skewness_", "_kurtosis_")):
        rel_tolerance = max(rel_tolerance, 1e-5)

    try:
        baseline_data = load_data_file(baseline_file)
        result_data = load_data_file(result_file)
        
        # Check shape compatibility
        if baseline_data.shape != result_data.shape:
            differences.append(f"Shape mismatch: baseline {baseline_data.shape} vs result {result_data.shape}")
            return differences
        
        # Calculate absolute differences
        abs_diff = np.abs(baseline_data - result_data)
        
        # Calculate relative differences
        # Handle case where baseline values might be zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = abs_diff / (np.abs(baseline_data) + 1e-16)
        
        # Determine adaptive absolute tolerance based on data magnitude
        # Exclude first two columns if they exist (likely time and volume)
        if baseline_data.shape[1] > 2:
            data_for_scaling = baseline_data[:, 2:]  # Skip first two columns
        else:
            data_for_scaling = baseline_data
        
        # Use the maximum magnitude in the dataset to scale absolute tolerance
        max_magnitude = np.max(np.abs(data_for_scaling))
        if is_on_github_actions() and ("elastic_strain" in baseline_file):
            # Currently running on 1 core leads to varying differences in the elastic strains
            adaptive_abs_tolerance = max(5e-6, max_magnitude * 1e-7)
        else:
            adaptive_abs_tolerance = max(abs_tolerance, max_magnitude * rel_tolerance)


        
        # A difference is acceptable if EITHER:
        # 1. Relative difference is below tolerance, OR  
        # 2. Absolute difference is below the adaptive absolute tolerance
        acceptable_diff = (rel_diff <= rel_tolerance) | (abs_diff <= adaptive_abs_tolerance)
        
        # Find locations where differences are NOT acceptable
        diff_locations = np.where(~acceptable_diff)
        
        if len(diff_locations[0]) > 0:
            # Report first few significant differences
            max_reports = min(10, len(diff_locations[0]))  # Limit to first 10 differences
            
            for i in range(max_reports):
                row, col = diff_locations[0][i], diff_locations[1][i]
                baseline_val = baseline_data[row, col]
                result_val = result_data[row, col]
                abs_diff_val = abs_diff[row, col]
                rel_diff_val = rel_diff[row, col]
                
                differences.append(
                    f"Row {row+2}, Col {col+1}: baseline={baseline_val:.6e}, "
                    f"result={result_val:.6e}, abs_diff={abs_diff_val:.6e}, "
                    f"rel_diff={rel_diff_val:.6e} (tol: rel={rel_tolerance:.1e}, abs={adaptive_abs_tolerance:.1e})"
                )
            
            if len(diff_locations[0]) > max_reports:
                differences.append(f"... and {len(diff_locations[0]) - max_reports} more differences")
                
            # Add summary of tolerance criteria
            differences.insert(0, f"Using adaptive absolute tolerance: {adaptive_abs_tolerance:.1e} "
                             f"(based on max data magnitude: {max_magnitude:.1e})")
        
    except Exception as e:
        differences.append(f"Error comparing files: {str(e)}")
    
    return differences

def validate_test_results(test_name: str, baseline_dir: str, result_dir: str) -> TestResult:
    """
    Validate test results by comparing all files in baseline vs result directories
    
    Args:
        test_name: Name of the test case
        baseline_dir: Directory containing baseline reference files
        result_dir: Directory containing test result files
        
    Returns:
        TestResult object with validation details
    """
    missing_files = []
    file_differences = {}
    
    try:
        if not os.path.exists(baseline_dir):
            return TestResult(test_name, False, [], {}, f"Baseline directory not found: {baseline_dir}")
        
        if not os.path.exists(result_dir):
            return TestResult(test_name, False, [], {}, f"Result directory not found: {result_dir}")
        
        # Get all files in baseline directory
        baseline_files = set()
        for root, _, files in os.walk(baseline_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), baseline_dir)
                baseline_files.add(rel_path)
        
        # Check for missing files in result directory
        for baseline_file in baseline_files:
            result_file_path = os.path.join(result_dir, baseline_file)
            if not os.path.exists(result_file_path):
                missing_files.append(baseline_file)
            else:
                # Compare files
                baseline_file_path = os.path.join(baseline_dir, baseline_file)
                differences = compare_files(baseline_file_path, result_file_path)
                if differences:
                    file_differences[baseline_file] = differences
        
        # Test passes if no missing files and no significant differences
        passed = len(missing_files) == 0 and len(file_differences) == 0
        
        return TestResult(test_name, passed, missing_files, file_differences)
        
    except Exception as e:
        return TestResult(test_name, False, [], {}, f"Validation error: {str(e)}")

def run_single_test(params: Tuple[str, str]) -> TestResult:
    """
    Run a single test case and validate results
    
    Args:
        params: Tuple of (toml_file, basename)
        
    Returns:
        TestResult object
    """
    toml_file, basename = params
    logger.info(f"Running test case: {toml_file} (basename: {basename})")
    
    try:
        # Get current working directory
        result = subprocess.run('pwd', stdout=subprocess.PIPE, text=True)
        pwd = result.stdout.strip()
        
        # Determine number of MPI processes
        if not is_on_github_actions():
            np_flag = '-np 2'
        else:
            np_flag = '-np 1'
        
        # Run the mechanics simulation
        cmd = f'mpirun {np_flag} {pwd}/../bin/mechanics -opt {toml_file}'
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              shell=True, text=True)
        
        if result.returncode != 0:
            error_msg = f"Simulation failed: {result.stderr}"
            logger.error(error_msg)
            return TestResult(basename, False, [], {}, error_msg)
        
        # Validate results
        baseline_dir = os.path.join(pwd, 'test_results', basename)
        result_dir = os.path.join(pwd, 'results', basename)
        
        test_result = validate_test_results(basename, baseline_dir, result_dir)
        
        # Log results
        if test_result.passed:
            logger.info(f"✓ Test {basename} PASSED")
        else:
            logger.error(f"✗ Test {basename} FAILED")
            if test_result.missing_files:
                logger.error(f"  Missing files: {test_result.missing_files}")
            for file, diffs in test_result.file_differences.items():
                logger.error(f"  Differences in {file}:")
                for diff in diffs[:3]:  # Show first 3 differences
                    logger.error(f"    {diff}")
        
        return test_result
        
    except Exception as e:
        error_msg = f"Test execution error: {str(e)}"
        logger.error(error_msg)
        return TestResult(basename, False, [], {}, error_msg)

def cleanup_result_files(test_cases: List[Tuple[str, str]], results_dir: str = "results"):
    """
    Clean up any existing result files before running tests
    
    Args:
        test_cases: List of (toml_file, basename) tuples
        results_dir: Directory containing result subdirectories
    """
    for _, basename in test_cases:
        result_subdir = os.path.join(results_dir, basename)
        if os.path.exists(result_subdir):
            logger.info(f"Cleaning up existing results in {result_subdir}")
            subprocess.run(f'rm -rf {result_subdir}', shell=True)

def run_all_tests() -> bool:
    """
    Discover and run all test cases in parallel
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Discover test cases
    test_cases = discover_test_cases()
    
    if not test_cases:
        logger.error("No test cases found!")
        return False
    
    # Clean up any existing result files
    cleanup_result_files(test_cases)
    
    # Determine number of parallel processes
    if platform == "linux" or platform == "linux2":
        num_processes = max(1, len(os.sched_getaffinity(0)) // 2)
    else:
        num_processes = max(1, multiprocessing.cpu_count() // 2)
    
    logger.info(f"Running {len(test_cases)} tests with {num_processes} parallel processes")
    
    # Run tests in parallel
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(run_single_test, test_cases)
    
    # Summarize results
    passed_tests = [r for r in results if r.passed]
    failed_tests = [r for r in results if not r.passed]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SUMMARY: {len(passed_tests)}/{len(results)} tests passed")
    logger.info(f"{'='*60}")
    
    if failed_tests:
        logger.error("\nFAILED TESTS:")
        for result in failed_tests:
            logger.error(f"\n❌ {result.test_name}:")
            if result.error_message:
                logger.error(f"   Error: {result.error_message}")
            if result.missing_files:
                logger.error(f"   Missing files: {', '.join(result.missing_files)}")
            for file, diffs in result.file_differences.items():
                logger.error(f"   Differences in {file}:")
                for diff in diffs[:3]:  # Limit output
                    logger.error(f"     {diff}")
                if len(diffs) > 3:
                    logger.error(f"     ... and {len(diffs)-3} more")
    
    return len(failed_tests) == 0

class TestExaConstit(unittest.TestCase):
    """Unit test wrapper for ExaConstit validation"""
    
    def test_all_cases(self):
        """Run all discovered test cases and validate results"""
        success = run_all_tests()
        self.assertTrue(success, "One or more ExaConstit tests failed")

if __name__ == '__main__':
    # If run directly, execute tests with detailed logging
    if len(os.sys.argv) == 1:
        # Run with detailed output
        success = run_all_tests()
        exit(0 if success else 1)
    else:
        # Run as unittest
        unittest.main()