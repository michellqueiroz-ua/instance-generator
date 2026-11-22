"""
Utilities for integrating C++ solver with the instance generator.
"""
import subprocess
import os
import json
import pandas as pd
from pathlib import Path
import time
import platform
import shutil


class SolverConfig:
    """Configuration for the C++ solver."""
    
    def __init__(self):
        # Path to compiled solver executable
        self.solver_path = self._detect_solver_path()
        
        # Default solver parameters
        self.time_limit = 3600  # 1 hour in seconds
        self.threads = 4
        
        # ODBRP-specific parameters
        self.seed = 0
        self.number_depots = 1
        self.depot_ids = [0]  # List of depot node IDs
        self.type_vehicles = 1
        self.number_vehicles1 = 5
        self.number_vehicles2 = 0
        self.number_vehicles3 = 0
        self.capacity_vehicles = [4]  # List of capacities per vehicle type
        self.init_temperature = 1.3
        self.lambda_param = 0.9
        self.maxnrep = 350
        self.increase_rep = 800
        self.total_requests = None  # Auto-detect from file
        
    def _detect_solver_path(self):
        """Detect solver executable based on platform."""
        possible_paths = [
            'solver/bin/a.out',           # Compiled from ODBRP_P
            'solver/bin/solver',          # Generic Linux
            'solver/bin/solver.exe',      # Generic Windows
            '../ODBRP_P/MDHODBRP-P/a.out', # Development path
            './a.out',
            './solver.exe',
            './solver',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None  # Solver not found
    
    def is_available(self):
        """Check if solver is available."""
        return self.solver_path is not None and os.path.exists(self.solver_path)


def compile_solver(progress_callback=None):
    """
    Automatically compile the ODBRP solver based on the current OS.
    
    Args:
        progress_callback: Function to call with progress messages
        
    Returns:
        dict: Result with 'status' ('success' or 'error') and 'message'
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)
    
    log("🔧 Starting automatic compilation...")
    
    # Check if g++ is available
    if not shutil.which('g++'):
        return {
            'status': 'error',
            'message': 'g++ compiler not found. Please install: Windows (MinGW-w64), Linux (build-essential), Mac (Xcode)'
        }
    
    # Find ODBRP source directory
    source_paths = [
        '../ODBRP_P/MDHODBRP-P',
        'C:/Users/micqu/Desktop/ODBRP_P/MDHODBRP-P',
        os.path.expanduser('~/Desktop/ODBRP_P/MDHODBRP-P'),
    ]
    
    source_dir = None
    for path in source_paths:
        if os.path.exists(path):
            source_dir = path
            break
    
    if not source_dir:
        return {
            'status': 'error',
            'message': 'ODBRP source code not found. Expected at: ../ODBRP_P/MDHODBRP-P/'
        }
    
    log(f"📂 Found source: {source_dir}")
    
    # Create solver/bin directory
    os.makedirs('solver/bin', exist_ok=True)
    
    # Compile based on OS
    system = platform.system()
    
    try:
        if system == 'Windows':
            log("🪟 Detected Windows - compiling with g++...")
            # Run compilation directly
            compile_cmd = [
                'g++', '-fopenmp', '-mcmodel=large', '-std=c++17', '-O2',
                '-o', 'solver/bin/a.out',
                os.path.join(source_dir, 'odbrp-parallel.cpp')
            ]
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                # Check if it's the common MinGW addressing issue
                if "too large for field" in result.stderr or "value of" in result.stderr:
                    return {
                        'status': 'error',
                        'message': (
                            'ODBRP solver cannot compile on Windows MinGW due to large array sizes. '
                            'This is a known MinGW limitation. Options:\n\n'
                            '1. ✅ Use WSL (Windows Subsystem for Linux) - Recommended\n'
                            '2. Use Docker with Linux container\n'
                            '3. Compile on a Linux machine\n'
                            '4. Use Visual Studio C++ compiler (requires setup)\n\n'
                            'Your instance files are ready and can be solved on Linux!'
                        )
                    }
                return {
                    'status': 'error',
                    'message': f'Compilation failed: {result.stderr}'
                }
            
            log("✅ Compilation successful!")
            return {
                'status': 'success',
                'message': 'Solver compiled successfully',
                'solver_path': 'solver/bin/a.out'
            }
            
        else:  # Linux or Mac
            log(f"🐧 Detected {system} - compiling with g++...")
            compile_cmd = [
                'g++', '-fopenmp', '-mcmodel=large', '-std=c++17', '-O2',
                '-o', 'solver/bin/a.out',
                os.path.join(source_dir, 'odbrp-parallel.cpp')
            ]
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                return {
                    'status': 'error',
                    'message': f'Compilation failed: {result.stderr}'
                }
            
            # Make executable
            os.chmod('solver/bin/a.out', 0o755)
            
            log("✅ Compilation successful!")
            return {
                'status': 'success',
                'message': 'Solver compiled successfully',
                'solver_path': 'solver/bin/a.out'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'status': 'error',
            'message': 'Compilation timed out after 120 seconds'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Compilation error: {str(e)}'
        }


def compile_solver_wsl(progress_callback=None):
    """
    Compile the ODBRP solver using WSL (for Windows with large array support).
    
    Args:
        progress_callback: Function to call with progress messages
        
    Returns:
        dict: Result with 'status' ('success' or 'error') and 'message'
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)
    
    log("🔧 Starting WSL compilation...")
    
    # Check if WSL is available
    if not shutil.which('wsl'):
        return {
            'status': 'error',
            'message': 'WSL not found. Please install Windows Subsystem for Linux.'
        }
    
    # Find ODBRP source directory
    source_paths = [
        'C:/Users/micqu/Desktop/ODBRP_P/MDHODBRP-P',
        '../ODBRP_P/MDHODBRP-P',
        os.path.expanduser('~/Desktop/ODBRP_P/MDHODBRP-P'),
    ]
    
    source_dir = None
    for path in source_paths:
        if os.path.exists(path):
            source_dir = os.path.abspath(path)
            break
    
    if not source_dir:
        return {
            'status': 'error',
            'message': 'ODBRP source code not found. Expected at: C:/Users/micqu/Desktop/ODBRP_P/MDHODBRP-P/'
        }
    
    log(f"📂 Found source: {source_dir}")
    
    # Convert Windows path to WSL path
    def win_to_wsl_path(path):
        path = os.path.abspath(path).replace('\\', '/')
        if ':' in path:
            drive, rest = path.split(':', 1)
            return f'/mnt/{drive.lower()}{rest}'
        return path
    
    wsl_source_dir = win_to_wsl_path(source_dir)
    wsl_output_dir = win_to_wsl_path(os.path.abspath('solver/bin'))
    
    # Create solver/bin directory
    os.makedirs('solver/bin', exist_ok=True)
    
    log("🐧 Compiling with WSL g++...")
    
    try:
        # Build WSL compilation command
        wsl_cmd = f'wsl bash -c "cd {wsl_source_dir} && g++ -fopenmp -mcmodel=large -std=c++17 -O2 odbrp-parallel.cpp -o a.out && cp a.out {wsl_output_dir}/a.out"'
        
        result = subprocess.run(wsl_cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return {
                'status': 'error',
                'message': f'WSL compilation failed: {result.stderr}'
            }
        
        log("✅ Compilation successful!")
        return {
            'status': 'success',
            'message': 'Solver compiled successfully with WSL',
            'solver_path': 'solver/bin/a.out'
        }
        
    except subprocess.TimeoutExpired:
        return {
            'status': 'error',
            'message': 'Compilation timed out after 120 seconds'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Compilation error: {str(e)}'
        }


def run_solver(instance_path, output_path, config=None, progress_callback=None, travel_time_path=None):
    """
    Run the C++ ODBRP solver on an instance.
    
    Args:
        instance_path: Path to instance file (CSV with requests)
        output_path: Path where solution should be saved
        config: SolverConfig object (optional)
        progress_callback: Function to call with progress updates
        travel_time_path: Path to travel time matrix (optional)
        
    Returns:
        dict: Results containing status, time, solution_path, etc.
    """
    if config is None:
        config = SolverConfig()
    
    # Auto-compile if solver not found
    if not config.is_available():
        if progress_callback:
            progress_callback("⚙️ Solver not found - attempting automatic compilation...")
        
        compile_result = compile_solver(progress_callback)
        
        if compile_result['status'] == 'error':
            return {
                'status': 'error',
                'message': f"Solver not found and compilation failed: {compile_result['message']}",
                'solver_path': config.solver_path
            }
        
        # Refresh solver path after compilation
        config.solver_path = compile_result['solver_path']
        
        if progress_callback:
            progress_callback("✅ Solver compiled successfully - starting optimization...")
    
    # Auto-detect total_requests from file if not set
    if config.total_requests is None:
        try:
            df = pd.read_csv(instance_path)
            config.total_requests = len(df)
        except:
            config.total_requests = 100  # Default fallback
    
    # Depot is station index 0 (first bus station in the transit network)
    # This is standard for ODBRP instances - depot must be a bus station, not a street node
    if config.depot_ids == [0]:  # If using default depot
        if progress_callback:
            progress_callback(f"📍 Using depot=0 (first bus station in transit network)")
    
    # Find travel time matrix if not provided
    if travel_time_path is None:
        instance_dir = os.path.dirname(os.path.dirname(instance_path))
        
        # Try multiple locations for travel time matrix
        possible_locations = [
            # 1. travel_time_matrix folder (standard location)
            (os.path.join(instance_dir, 'travel_time_matrix'), ['travel_time', '.csv']),
            # 2. csv folder with .tt.drive.csv pattern (actual location)
            (os.path.join(instance_dir, 'csv'), ['.tt.drive.csv']),
            # 3. Same directory as instance file
            (os.path.dirname(instance_path), ['.tt.drive.csv', 'travel_time']),
        ]
        
        for search_dir, patterns in possible_locations:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith('.csv') and any(p in file for p in patterns):
                        travel_time_path = os.path.join(search_dir, file)
                        if progress_callback:
                            progress_callback(f"📍 Found travel time matrix: {file}")
                        break
            if travel_time_path:
                break
    
    if travel_time_path is None or not os.path.exists(travel_time_path):
        return {
            'status': 'error',
            'message': 'Travel time matrix not found. Generate the instance with network data first.',
        }
    
    # Build command for ODBRP solver
    # Instance type: 1=festival, 2=commuting, 3=nightlife/concert (simpler, no dual-times)
    cmd = [
        config.solver_path,
        '--filename_requests', '3', instance_path,  # Type 3 for basic ODBRP instances
        '--seed', str(config.seed),
        '--filename_travel_time', travel_time_path,
        '--output_file', output_path,
        '--number_depots', str(config.number_depots),
    ]
    
    # Add depot IDs
    cmd.append('--depot')
    for depot_id in config.depot_ids[:config.number_depots]:
        cmd.append(str(depot_id))
    
    # Add vehicle configuration
    cmd.extend([
        '--type_vehicles', str(config.type_vehicles),
        '--number_vehicles1', str(config.number_vehicles1),
        '--number_vehicles2', str(config.number_vehicles2),
        '--number_vehicles3', str(config.number_vehicles3),
    ])
    
    # Add vehicle capacities
    cmd.append('--capacity_vehicles')
    for cap in config.capacity_vehicles:
        cmd.append(str(cap))
    
    # Add algorithm parameters
    cmd.extend([
        '--init_temperature', str(config.init_temperature),
        '--lamba', str(config.lambda_param),  # Note: typo in original solver code
        '--maxnrep', str(config.maxnrep),
        '--increase_rep', str(config.increase_rep),
        '--total_requests', str(config.total_requests),
    ])
    
    if progress_callback:
        progress_callback("Starting ODBRP solver...")
    
    start_time = time.time()
    
    # Check if we're on Windows and if solver is a Linux executable (from WSL)
    is_wsl_executable = False
    if platform.system() == 'Windows':
        # Check if executable is ELF (Linux) format
        try:
            with open(config.solver_path, 'rb') as f:
                magic = f.read(4)
                if magic == b'\x7fELF':  # ELF magic number
                    is_wsl_executable = True
                    if progress_callback:
                        progress_callback("Detected WSL Linux executable - running through WSL...")
        except:
            pass
    
    # If WSL executable, wrap command with wsl and convert paths
    if is_wsl_executable:
        # Convert Windows paths to WSL paths (/mnt/c/...)
        def win_to_wsl_path(path):
            path = os.path.abspath(path)
            # C:\Users\... -> /mnt/c/Users/...
            if len(path) > 1 and path[1] == ':':
                drive = path[0].lower()
                rest = path[2:].replace('\\', '/')
                return f'/mnt/{drive}{rest}'
            return path.replace('\\', '/')
        
        wsl_solver_path = win_to_wsl_path(config.solver_path)
        wsl_instance_path = win_to_wsl_path(instance_path)
        wsl_ttm_path = win_to_wsl_path(travel_time_path)
        wsl_output_path = win_to_wsl_path(output_path)
        
        # Build WSL command - use bash -c with proper quoting for paths with spaces
        bash_cmd = f'"{wsl_solver_path}" --filename_requests 3 "{wsl_instance_path}" --seed {config.seed} --filename_travel_time "{wsl_ttm_path}" --output_file "{wsl_output_path}" --number_depots {config.number_depots}'
        bash_cmd += ' --depot'
        for depot_id in config.depot_ids[:config.number_depots]:
            bash_cmd += f' {depot_id}'
        bash_cmd += f' --type_vehicles {config.type_vehicles} --number_vehicles1 {config.number_vehicles1} --number_vehicles2 {config.number_vehicles2} --number_vehicles3 {config.number_vehicles3}'
        bash_cmd += ' --capacity_vehicles'
        for cap in config.capacity_vehicles:
            bash_cmd += f' {cap}'
        bash_cmd += f' --init_temperature {config.init_temperature} --lamba {config.lambda_param} --maxnrep {config.maxnrep} --increase_rep {config.increase_rep} --total_requests {config.total_requests}'
        
        wsl_cmd = ['wsl', 'bash', '-c', bash_cmd]
        cmd = wsl_cmd
    
    try:
        # Create debug log file
        debug_log = os.path.join(os.path.dirname(output_path), 'solver_debug.log')
        with open(debug_log, 'w') as f:
            f.write(f"=== ODBRP Solver Debug Log ===\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if is_wsl_executable and len(cmd) >= 4:
                # Show the actual bash command for WSL
                f.write(f"Command: {cmd[0]} {cmd[1]} {cmd[2]} {cmd[3]}\n\n")
            else:
                f.write(f"Command: {' '.join(cmd)}\n\n")
        
        # Run solver
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.time_limit + 60  # Give extra buffer
        )
        
        elapsed_time = time.time() - start_time
        
        # Log output
        with open(debug_log, 'a') as f:
            f.write(f"\n=== Return Code: {result.returncode} ===\n")
            f.write(f"Elapsed Time: {elapsed_time:.2f}s\n\n")
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n\n")
        
        if result.returncode == 0:
            # Success
            if progress_callback:
                progress_callback(f"Solver completed in {elapsed_time:.2f}s")
            
            return {
                'status': 'success',
                'time': elapsed_time,
                'solution_path': output_path,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd)
            }
        else:
            # Solver failed
            return {
                'status': 'error',
                'message': f'Solver exited with code {result.returncode}. Check {debug_log} for details.',
                'time': elapsed_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd),
                'debug_log': debug_log
            }
            
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'message': f'Solver exceeded time limit of {config.time_limit}s',
            'time': config.time_limit
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to run solver: {str(e)}',
            'exception': str(e)
        }


def parse_solution(solution_path):
    """
    Parse solver output file.
    
    Args:
        solution_path: Path to solution file
        
    Returns:
        dict: Parsed solution with routes, statistics, etc.
    """
    if not os.path.exists(solution_path):
        return None
    
    # MODIFY THIS based on your solver's output format
    # Example assumes CSV format with routes
    
    try:
        # Try CSV format
        if solution_path.endswith('.csv'):
            df = pd.read_csv(solution_path)
            
            # Extract basic statistics
            solution = {
                'format': 'csv',
                'data': df,
                'statistics': {
                    'num_routes': df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else 0,
                    'num_requests_served': len(df),
                    'total_distance': df['distance'].sum() if 'distance' in df.columns else 0,
                    'total_time': df['time'].sum() if 'time' in df.columns else 0,
                }
            }
            
            return solution
            
        # Try JSON format
        elif solution_path.endswith('.json'):
            with open(solution_path, 'r') as f:
                solution = json.load(f)
            solution['format'] = 'json'
            return solution
            
        else:
            # Unknown format, read as text
            with open(solution_path, 'r') as f:
                content = f.read()
            return {
                'format': 'text',
                'content': content
            }
            
    except Exception as e:
        return {
            'error': f'Failed to parse solution: {str(e)}'
        }


def extract_routes(solution):
    """
    Extract vehicle routes from solution.
    
    Args:
        solution: Parsed solution dict
        
    Returns:
        list: List of routes, each route is a list of stops
    """
    if solution is None or 'data' not in solution:
        return []
    
    df = solution['data']
    
    # MODIFY THIS based on your solution format
    # Example: group by vehicle_id and order by sequence
    if 'vehicle_id' in df.columns:
        routes = []
        for vehicle_id in df['vehicle_id'].unique():
            vehicle_df = df[df['vehicle_id'] == vehicle_id]
            
            # Order by sequence if available
            if 'sequence' in vehicle_df.columns:
                vehicle_df = vehicle_df.sort_values('sequence')
            
            route = vehicle_df.to_dict('records')
            routes.append({
                'vehicle_id': vehicle_id,
                'stops': route
            })
        
        return routes
    
    return []


def format_statistics(solution):
    """
    Format solution statistics for display.
    
    Args:
        solution: Parsed solution dict
        
    Returns:
        dict: Formatted statistics
    """
    if solution is None:
        return {}
    
    if 'statistics' in solution:
        return solution['statistics']
    
    # Try to compute basic stats
    if 'data' in solution:
        df = solution['data']
        return {
            'rows': len(df),
            'columns': len(df.columns),
        }
    
    return {}
