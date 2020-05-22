#!/bin/bash

if [ "$#" -ne 2 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 project[dummy|example|spacy|learning_fix] model[preprocess|train|predict|val] " >&2
  exit 1
fi
CurrentDate=$(date +%F)

#ProjectName="spacy"
#ProjectName="dummy"
#ProjectName="example"
#ProjectName="learning_fix"
ProjectName=$1

# preprocess|train|predict|val
model=$2

# Root envs
export RootPath=`pwd`
export PYTHONPATH=${PYTHONPATH}:${RootPath}

ProjectBechmarks=${RootPath}/benchmarks/${ProjectName}


# Project envs
ProjectData=${ProjectBechmarks}/data

# Processed data folder to save intermediate data
ProjectProcessedDataDir=${ProjectBechmarks}/processed

# Default config file
ProjectConfig=${ProjectBechmarks}/configs/default_config.json

# Default logs file
ProjectLog=${ProjectBechmarks}/logs/${model}-${CurrentDate}.log

# Default Checkpoint file
ProjectCheckpoint=${ProjectBechmarks}/checkpoints/checkpoint-${ProjectName}.pth

# choices=['default', 'sequence', 'tree', 'path'], default='tree'
Positional_Style="default"

case ${model} in
  "abstract")
    set -x
    export JAVA_OPTS="-Xmx32G -Xms1g -Xss512M"
    scala bin/java_abstract-1.0-jar-with-dependencies.jar ${RootPath}/conf/application.conf
  ;;
  "preprocess")
      set -x
      python "${ProjectBechmarks}"/preprocess.py \
                              --project_name="${ProjectName}" \
                              --project_config="${ProjectConfig}" \
                              --project_raw_dir="${ProjectData}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --debug=True \
                              --position_style=${Positional_Style} \
                              --phase="${model}"
  ;;
  "train")
      set -x
      python "${ProjectBechmarks}"/train.py \
                                      --project_name="${ProjectName}" \
                                      --project_config="${ProjectConfig}" \
                                      --project_raw_dir="${ProjectData}" \
                                      --project_processed_dir="${ProjectProcessedDataDir}" \
                                      --project_log="${ProjectLog}" \
                                      --project_checkpoint="${ProjectCheckpoint}" \
                                      --debug=False \
                                      --phase="${model}" \
                                      --device='cuda' \
                                      --position_style=${Positional_Style} \
                                      --device_id=[1]

  ;;
  "predict")
      set -x
      python "${ProjectBechmarks}"/predict.py \
                              --project_name="${ProjectName}" \
                              --project_config="${ProjectConfig}" \
                              --project_raw_dir="${ProjectData}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --debug=False \
                              --phase="${model}" \
                              --device='cuda' \
                              --device_id=[1] \
                              --position_style=${Positional_Style} \
                              --source="private TYPE_1 getType ( TYPE_2 VAR_1 ) { TYPE_3 VAR_2 = new TYPE_3 ( STRING_1 ) ; return new TYPE_1 ( VAR_2 , VAR_2 ) ; }"

  ;;
  "val")
      set -x
      python "${ProjectBechmarks}"/evaluate.py \
                              --project_name="${ProjectName}" \
                              --project_config="${ProjectConfig}" \
                              --project_raw_dir="${ProjectData}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --debug=False \
                              --phase="${model}" \
                               --device='cuda' \
                              --device_id=[1] \
                              --position_style=${Positional_Style} \
                              --save_result="${ProjectProcessedDataDir}"/"${ProjectName}"_eval.txt
  ;;
   *)
     echo "There is no match case for ${model}"
     echo "Usage: $0 project[dummy|example|spacy|learning_fix] model[preprocess|train|predict|val] " >&2
     exit 1
  ;;
esac
