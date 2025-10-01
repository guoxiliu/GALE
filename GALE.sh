#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 100GB
#SBATCH --mail-type NONE
#SBATCH --time 1:00:00

# Add --ntasks <#CPUS> --gpus <model>:1 flags to sbatch of this script

# Configuration
override=0; # Set to 1 for override
n_repeats=3; # Times to repeat each dataset
SFCP="_sfcp"; # Set "_sfcp" to use integrated kernel, set "" to use GALE kernels

cd /home/tlranda/TetraTopo_GPU/GALE_Partition;
source /home/tlranda/TetraTopo_GPU/env.sh; # Modulefiles and list

# Set number of CPUs once
ncpus=$( lscpu | grep -e "^CPU(s):" | awk '{print $NF}' );
# GALE should use #(logical cores minus 2) as threads for best performance
full_subscribe=$((${ncpus}-2));
echo "Execute on host ${HOSTNAME}";
nvidia-smi;
echo "NCPUS: ${ncpus}";
echo "Full subscribe: ${full_subscribe}";

outputdir="${HOSTNAME}${SFCP}_outputs";
mkdir -p ${outputdir};

if [[ "${SFCP}" == "" ]]; then
    binary="ttkScalarFieldCriticalPoints";
else
    binary="criticalPointsKernel";
fi;

datasets=$( ls -d datasets/*.vtu );
declare -A target_array=( #[datasets/Bucket_100.vtu]=Result
                          #[datasets/viscousFingering_100.vtu]=concentration
                          #[datasets/ctBones_100.vtu]=Scalars_
                          #[datasets/Engine_100.vtu]=Scalars_
                          #[datasets/Foot_100.vtu]=Scalars_
                          #[datasets/Fish_100.vtu]=Elevation
                          #[datasets/Asteroid_100.vtu]=scalar
                          #[datasets/Hole_100.vtu]=Result   # -b 32
                          [datasets/Stent_100.vtu]=Scalars_
                         );
for ds in ${datasets[@]}; do
    echo "${ds}";
    shortname=$( basename ${ds} | tr "." "\n" | head -n 1 );
    # Determine the dataset's proper array to target
    used_array="${target_array[$ds]}";
    if [[ ! ${used_array+_} ]]; then
        # Auto-lookup for attached arrays, filtered down to hopefully just the PointData segment excluding any array named "_index"
        found_arrays=$( ./build_${HOSTNAME}/bin/ttkScalarFieldCriticalPoints -i ${ds} -l | sed -n '/.*- \w/,/.*CellData:/p' | awk '{print $NF}' | grep -v -e "CellData:" -e "_index" | sort | uniq );
        # Default to picking the first array from the list
        used_array=$( echo ${found_arrays} | tr " " "\n" | head -n 1 );
        echo "No array set for ${ds}; found arrays: ${found_arrays}";
    fi;
    if [[ "${used_array}" == "" ]]; then
        echo -e "\t! No array; skipping !";
        continue;
    fi;
    # Array selected; proceed
    echo -e "\tUsing array: ${used_array}";
    cmd="./build_${HOSTNAME}/bin/${binary} -i ${ds} -a ${used_array} -t ${full_subscribe}";
    for iteration in `seq 1 ${n_repeats}`; do
        to_make="${outputdir}/${shortname}_${full_subscribe}CPUS_iter_${iteration}.output";
	if [[ ! -e "${to_make}" || ${override} == 1 ]]; then
	    full_command="${cmd} > ${to_make}";
	    echo "${full_command}";
            eval "${full_command}";
	else
	    echo "${to_make} exists, skipping...";
	fi;
    done;
done;
# Analyze all captured traces
python3 processGALE.py ${outputdir}/*.output;

