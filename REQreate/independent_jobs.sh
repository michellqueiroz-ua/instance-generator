#!/bin/bash
#SBATCH --job-name=ExperimentsPaper5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michell.queiroz@uantwerpen.be
#SBATCH -o Output_%j.out
#SBATCH -e Output_%j.err
#SBATCH --partition=skylake

module load calcua/supported
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh
conda activate ox

# submit_all.sh - Submit jobs for all instances

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Script directory: $SCRIPT_DIR"
echo "Current directory: $(pwd)"
echo "Looking for instances_parallel folders..."
ls -d instances_parallel/*/ 2>/dev/null || echo "No folders found in instances_parallel"
echo ""

# Function to determine filename_requests parameter based on instance type
get_filename_requests() {
    local folder_name=$1
    case $folder_name in
        festival)
            echo "1"
            ;;
        commuting|commuting2)
            echo "2"
            ;;
        concert|concert2)
            echo "3"
            ;;
        nightlife|nightlife2)
            echo "3"
            ;;
        *)
            echo "1"  # default
            ;;
    esac
}

# Function to get output prefix based on instance type
get_output_prefix() {
    local folder_name=$1
    case $folder_name in
        festival)
            echo "R1_festival"
            ;;
        commuting|commuting2)
            echo "R1_commuting"
            ;;
        concert|concert2)
            echo "R1_concert"
            ;;
        nightlife|nightlife2)
            echo "R1_nightlife"
            ;;
        *)
            echo "R1_${folder_name}"
            ;;
    esac
}

# Initialize counter for seed generation (same logic as mdhodbrp-atools.sh)
counterx=6

# Number of cores to use
ncores=1

# Loop through all subfolders in instances_parallel
for folder in instances_parallel/*/; do
    echo "Processing folder: $folder"
    
    # Get folder name without trailing slash
    jobname=$(basename "${folder%/}")
    
    # Get appropriate parameters for this instance type
    filename_requests=$(get_filename_requests "$jobname")
    output_prefix=$(get_output_prefix "$jobname")
    
    # Get all CSV files in this folder
    files=("$folder"*.csv)
    
    # Process each file individually
    for input_file in "${files[@]}"; do
        # Get basename for output file naming
        file_basename=$(basename "$input_file" .csv)
        
        # Get absolute path for input file
        abs_input_file="$(cd "$(dirname "$input_file")" && pwd)/$(basename "$input_file")"
        
        # Run 10 iterations with different seeds (same as mdhodbrp-atools.sh)
        for iter in {1..10}; do
            # Generate seed (same as mdhodbrp-atools.sh)
            counterx=$((counterx+1))
            seed1=111
            seed1=$(($seed1 + $counterx))
            
            # Debug output
            echo "Submitting job for: $input_file (iteration $iter, seed $seed1)"
            
            # Submit independent LSF job for each file+seed combination
            bsub -J "MDHODBRP_${jobname}_${file_basename}_s${seed1}" \
                 -q hpc \
                 -n $ncores \
                 -R "span[hosts=1]" \
                 -R "rusage[mem=8GB]" \
                 -M 9GB \
                 -W 24:00 \
                 -o "${SCRIPT_DIR}/${output_prefix}_${file_basename}_s${seed1}.out" \
                 -e "${SCRIPT_DIR}/${output_prefix}_${file_basename}_s${seed1}.err" \
                 "cd ${SCRIPT_DIR} && ./a.out --filename_requests \"${filename_requests}\" \"${abs_input_file}\" --seed ${seed1} --filename_travel_time \"${SCRIPT_DIR}/travel_time_updated3.csv\" --output_file \"${SCRIPT_DIR}/${output_prefix}_results.txt\" --number_depots 3 --depot 5825 5826 5827 --type_vehicles 3 --number_vehicles1 15 --number_vehicles2 8 --number_vehicles3 0 --capacity_vehicles 4 8 12 --init_temperature 1.3 --lamba 0.9 --maxnrep 350 --increase_rep 800 --total_requests 900"
        done
    done
done