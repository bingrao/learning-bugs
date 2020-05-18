#!/bin/bash

if [ "$#" -ne 2 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 project[dummy|example|spacy|learning_fix] model[abstract|preprocess|train|predict|eval] " >&2
  exit 1
fi
CurrentDate=$(date +%F)

#ProjectName="spacy"
#ProjectName="dummy"
#ProjectName="example"
#ProjectName="learning_fix"
ProjectName=$1

# abstract|preprocess|train|predict|eval
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

case ${model} in
  "abstract")
    set -x
    mvn assembly:assembly
    scala -Dlog4j.configuration=src/main/resources/log4j.properties target/learning-bugs-1.0-SNAPSHOT-jar-with-dependencies.jar ${RootPath}/src/main/resources/application.conf
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
                              --debug=False \
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
                              --source="There is an imbalance here ."

  ;;
  "eval")
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
                              --save_result="${ProjectProcessedDataDir}"/"${ProjectName}"_eval.txt
  ;;
   *)
     echo "There is no match case for ${model}"
     echo "Usage: $0 project[dummy|example|spacy|learning_fix] model[preprocess|train|predict|eval] " >&2
     exit 1
  ;;
esac