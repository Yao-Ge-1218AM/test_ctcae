#!/usr/bin/env bash
set -euo pipefail


#!/usr/bin/env bash
set -euo pipefail

##########################
# User-configurable vars #
##########################

# TOTAL number of partitions (i.e. number of parallel “chunks” to split your dataset into)
NUM_PARTITIONS=2

# TOTAL number of GPUs available on this machine
# NUM_GPUS=4
DESIGNATED_GPUS=(0 1 2 3 4 5)

# Derive the count of GPUs
NUM_GPUS=${#DESIGNATED_GPUS[@]}

PROJECT_DIR="/data/yangy39/llm_privacy"
MY_DIR="/netmnt/vast01/cbb01/lulab/yangy39/RLAIF"

SM_MODELS=(
  # "base"
  # "./models/qwen3-4b_PLH_gptData_v1/Beta2"
  # "meta-llama/Llama-3.2-3B-Instruct"
  # "Qwen/Qwen3-4B"
  # "./models/llama32_3b_PLH_gptData_v1/Beta0"
  # "./models/llama32_3b_PLH_gptData_v1/Beta0.5"
  # "./models/llama32_3b_PLH_gptData_v6/Beta4"
  # "./models/llama32_3b_PLH_gptData_v6/Beta2"
  # "./models/llama32_3b_PLH_gptData_v4/Beta4"
  # "./models/llama32_3b_PLH_gptData_v4/Beta2"
  # "./models/llama32_3b_PLH_gptData_v5/Beta4"
  "${PROJECT_DIR}/models/llama32_3b_PLH_gptData_v5/Beta2"
  # "./models/llama32_3b_PLH_gptData_v7_temp/Beta2"
  
)

# Your CM model paths
CM_MODELS=(
  # "./models/activation_xlm-roberta-large_v2"
  # "./models/activation_Qwen3-1.7B_v2"
  # "./models/activation_llama32_1b_v3"
  "${PROJECT_DIR}/models/activation_llama32_1b_v3"

)

# α, β, k values (only used when SM ≠ “base”)
# ALPHAS=(2 1)
# BETAS=(0.2 0.4 0.6 0.8 0.9 0.95)
# KS=(50000 500 20000)
# THRS=(0.9)

ALPHAS=(1)
BETAS=(0.2)
KS=(2)
THRS=(0.7)

SUFFIX="debug"
# SUFFIX="comparison"
# combination strategies
# COMBINATIONS=("comparison")
COMBINATIONS=("softmax_top_k")

# BM_MODEL and TEST_FILE remain as before
# BM_MODEL_NAME="PHI_llama33_70B_v3"
BM_MODEL_NAME="PHI_llama31_8B_v2"
BM_MODEL="${PROJECT_DIR}/models/${BM_MODEL_NAME}"
# BM_MODEL_NAME="Qwen3-32B"
# BM_MODEL="Qwen/${BM_MODEL_NAME}"
TEST_FILE="HealthBench"


DESIRED=(
  "${PROJECT_DIR}/models/llama32_3b_PLH_gptData_v5/Beta2_a1_b0.1_k500"
)


# make a little helper to join cm_key,α,β,k


##############################
# Compute GPUs per partition #
##############################

GPUS_PER_PART=$(( NUM_GPUS / NUM_PARTITIONS ))    # floor(G/P)
REMAINDER=$(( NUM_GPUS %  NUM_PARTITIONS ))       # leftover GPUs

# Given a 0-based partition index, return a comma-separated list of slots
gpu_list_for_partition() {
  local idx=$1

  if (( NUM_GPUS >= NUM_PARTITIONS )); then
    # multiple‐GPUs per partition
    local count=$GPUS_PER_PART
    if (( idx == NUM_PARTITIONS - 1 )) && (( REMAINDER > 0 )); then
      count=$(( GPUS_PER_PART + REMAINDER ))
    fi

    # compute slice start
    local start=$(( idx * GPUS_PER_PART ))
    # extract that many entries from the DESIGNATED_GPUS array
    local slice=( "${DESIGNATED_GPUS[@]:start:count}" )

    IFS=','; echo "${slice[*]}"; unset IFS

  else
    # fewer GPUs than partitions => one GPU per partition in round-robin
    local gpu_idx=$(( idx % NUM_GPUS ))
    echo "${DESIGNATED_GPUS[gpu_idx]}"
  fi
}

#######################
# Main benchmarking   #
#######################
for thr in "${THRS[@]}"; do
  for combination in "${COMBINATIONS[@]}"; do
    RESULTS_DIR="${MY_DIR}/results/${BM_MODEL_NAME}_${TEST_FILE}_${combination}_thr${thr}_${SUFFIX}"
    mkdir -p "${RESULTS_DIR}"
  
    for sm in "${SM_MODELS[@]}"; do
      sm_key="${sm#${PROJECT_DIR}/models/}"
      sm_key="${sm_key//\//.}"
  
      for cm in "${CM_MODELS[@]}"; do
        cm_key="${cm#${PROJECT_DIR}/models/}"
        cm_key="${cm_key//\//.}"
  
        # --------
        # Case 1: SM = "base" → no α/β/k loops
        # --------
        if [[ "${sm_key}" == "base" ]]; then
          # RESULTS_DIR="./results/Llama31_8B_baseline_${TEST_FILE}"
          for (( part=1; part<=NUM_PARTITIONS; part++ )); do
            # Compute 0-based index
            idx=$(( part - 1 ))
            GPU_LIST=$(gpu_list_for_partition "$idx")
  
            base_name="${RESULTS_DIR}/results_SM_${sm_key}_part${part}"
            part_file="${base_name}.json"
            if [[ -f "${part_file}" ]]; then
              echo "Skipping SM=${sm_key}, partition=${part} — exists: ${part_file}"
              continue
            fi
  
            echo "→ SM=${sm_key}, partition=${part}, GPUs=${GPU_LIST}"
              CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
              python ${MY_DIR}/benchmark_processor.py \
                -bm "${BM_MODEL}" \
                -sm "${sm}" \
                -cm "${cm}" \
                -thr "${thr}" \
                -c "${combination}" \
                -f "${TEST_FILE}" \
                -o "${base_name}.json" \
                -p "${part}" \
                -M "${NUM_PARTITIONS}" &
          done
  
          wait
          continue
          # RESULTS_DIR="./results/Llama31_8B_${TEST_FILE}_${combination}_thr0.7_sample"
        fi
  
        # -----------------------------------------
        # Case 2: SM ≠ "base" → loop over α, β, k
        # -----------------------------------------
        for alpha in "${ALPHAS[@]}"; do
          for beta in "${BETAS[@]}"; do
            for k in "${KS[@]}"; do


              current="${sm}_a${alpha}_b${beta}_k${k}"
              # fast membership test
              # if [[ ! " ${DESIRED[*]} " =~ " ${current} " ]]; then
              #   continue
              # fi
              
              # Adjust naming if combination does/doesn't include "top_k"
              if [[ "${combination}" == *"top_k"* ]]; then
                suffix="_CM_${cm_key}_a${alpha}_b${beta}_k${k}"
              else
                suffix="_CM_${cm_key}_a${alpha}_b${beta}"
              fi
  
              for (( part=1; part<=NUM_PARTITIONS; part++ )); do
                idx=$(( part - 1 ))
                GPU_LIST=$(gpu_list_for_partition "$idx")
  
                base_name="${RESULTS_DIR}/results_SM_${sm_key}${suffix}_part${part}"
                part_file="${base_name}_part${part}.json"
                echo "${part_file}"
  
                echo "→ combination=${combination} SM=${sm_key}, CM=${cm_key}, α=${alpha}, β=${beta}, k=${k}, partition=${part}, GPUs=${GPU_LIST}"
                start=$SECONDS
  
                 CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
                  python ${MY_DIR}/benchmark_processor.py \
                    -bm "${BM_MODEL}" \
                    -sm "${sm}" \
                    -cm "${cm}" \
                    -thr "${thr}" \
                    -a "${alpha}" \
                    -b "${beta}" \
                    -c "${combination}" \
                    -k "${k}" \
                    -f "${TEST_FILE}" \
                    -o "${base_name}.json" \
                    -p "${part}" \
                    -M "${NUM_PARTITIONS}" & \
                # echo "   Partition ${part} done in $(( SECONDS - start ))s" &
              done
  
              wait
            done
          done
        done
  
      done
    done
  done
done

