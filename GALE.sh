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

cd /home/tlranda/TetraTopo_GPU/GALE_SFCP;
source /home/tlranda/TetraTopo_GPU/env.sh; # Modulefiles and list

# Set number of CPUs once
ncpus=$( lscpu | grep -e "^CPU(s):" | awk '{print $NF}' );
# GALE should use #(logical cores minus 2) as threads for best performance
full_subscribe=$((${ncpus}));
echo "Execute on host ${HOSTNAME}";
nvidia-smi;
ngpus=1;
echo "NCPUS: ${ncpus}";
echo "Full subscribe: ${full_subscribe}";

outputdir="${HOSTNAME}_outputs";
mkdir -p ${outputdir};

binary="./build_2026/bin/criticalPointsKernel";

datasets=$( ls -d datasets/*_1000b.vtu );
declare -A target_array=( [datasets/Bucket_1000b.vtu]=Result
                          [datasets/viscousFingering_1000b.vtu]=concentration
                          [datasets/ctBones_1000b.vtu]=Scalars_
                          [datasets/Engine_1000b.vtu]=Scalars_
                          [datasets/Foot_1000b.vtu]=Scalars_
                          [datasets/Fish_1000b.vtu]=Elevation
                          [datasets/Asteroid_1000b.vtu]=scalar
                          [datasets/Hole_1000b.vtu]=Result   # -b 32
                          [datasets/Stent_1000b.vtu]=Scalars_
                         );
declare -A known_subscribe=( [datasets/Bucket_1000b.vtu]=0
                             [datasets/viscousFingering_1000b.vtu]=16
                             [datasets/Engine_1000b.vtu]=24 # 0 on TitanXP, which ==24
                             [datasets/Foot_1000b.vtu]=9
                             [datasets/Fish_1000b.vtu]=7
                             [datasets/Asteroid_1000b.vtu]=3 # Too long to compute
                             [datasets/ctBones_1000b.vtu]=1 # Too long to compute
                             [datasets/Hole_1000b.vtu]=3 # Too long to compute
                             [datasets/Stent_1000b.vtu]=1 # Too long to compute
                            );
skips=(
        # Too long to execute list
        # "datasets/Fish_1000b.vtu" # -- only too long on TitanXP
        #"datasets/Asteroid_1000b.vtu"
        "datasets/Hole_1000b.vtu"
        "datasets/Stent_1000b.vtu"
        "datasets/ctBones_1000b.vtu"
        # Not in final dataset list
        "datasets/Lobster_1000b.vtu"
        "datasets/Redsea_1000b.vtu"
        "datasets/Shapes_1000b.vtu"
    );
for ds in ${datasets[@]}; do
    skip_check=$( printf '%s\n' "${skips[@]}" | grep -Fx -- "${ds}" );
    if [[ "${skip_check}" == "${ds}" ]]; then
        echo "Skipping dataset ${ds} in skip array";
        continue;
    fi
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
    cmd="timeout 5m ${binary} -i ${ds} -a ${used_array} -t ${full_subscribe} -p _index";
    # Max threads reset PER dataset unless known
    max_threads="${known_subscribe[$ds]}";
    if [[ ${max_threads} -eq 0 ]]; then
        max_threads="${full_subscribe}";
    fi;
    echo "Set max threads == ${max_threads}";
    timed_out_gpu="";
    for gpu_count in `seq 0 $ngpus`; do
        gpu_count=$(( 2 ** ${gpu_count} ));
        base_cmd="${cmd} -g ${gpu_count}";
        for iteration in `seq 1 ${n_repeats}`; do
            to_make="${outputdir}/${shortname}_${gpu_count}_GPUS_${full_subscribe}CPUS_iter_${iteration}.output";
            timed_out_iteration="";
            if [[ ! -e "${to_make}" || ${override} == 1 ]]; then
                # Determine max threads
                while [[ ${max_threads} -gt 0 ]]; do
                    full_command="${base_cmd} -u ${max_threads} > ${to_make}";
                    echo "${full_command}";
                    eval "${full_command}";
                    rcode=$?;
                    if [[ ${rcode} -eq 0 ]]; then
                        break;
                    else
                        if [[ ${rcode} -eq 124 ]]; then
                            timed_out_iteration="Yes";
                            break;
                        else
                            echo "Unable to run at ${max_threads}, reducing thread count by 1";
                            max_threads=$(( ${max_threads}-1 ));
                        fi;
                    fi;
                done;
                if [[ "${timed_out_iteration}" == "Yes" ]]; then
                    echo "Time expired; skipping subsequent iterations";
                    timed_out_gpu="Yes";
                    break;
                fi;
            else
                echo "${to_make} exists, skipping...";
            fi;
        done;
        #if [[ "${timed_out_gpu}" == "Yes" ]]; then
        #    echo "Time expired; skipping subsequent GPU counts";
        #    break;
        #fi;
    done;
done;
# Analyze all captured traces
python3 processGALE.py ${outputdir}/*.output;

