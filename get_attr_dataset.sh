#!/bin/bash
MODELS=("FCN" "InceptionTime" ) # "FCN"
#DATASETS=( "GunPoint" "Chinatown" "ECG200" "DistalPhalanxOutlineCorrect" "PowerCons" "Earthquakes"  "Strawberry" "FordA" "Yoga" "Wafer" "HandOutlines") # "GunPoint" "Chinatown" "ECG200" "DistalPhalanxOutlineCorrect"
#"Computers" "Chinatown" "ECG200" "DistalPhalanxOutlineCorrect" "Yoga" "GunPoint" "PowerCons" "Earthquakes" "FordA" "Strawberry" "Wafer" "HandOutlines"
DATASETS=( "Coffee" "Chinatown" "Computers" "ECGFiveDays" "TwoLeadECG" "Lightning2"  "GunPointMaleVersusFemale")
XAI_NAMES=( "DeepLift" "GuidedBackprop" "InputXGradient" "IntegratedGradients" "KernelShap" "Lime" "Occlusion" "Saliency" ) # "DeepLift"
for dataset in "${DATASETS[@]}"; do
  for xai_name in "${XAI_NAMES[@]}"; do
      if [ "$xai_name" == "DeepLift" ]; then
          MODELS=("FCN")
      else
          MODELS=("FCN" "InceptionTime" )
      fi
      for model in "${MODELS[@]}"; do
          python get_datasets_attr.py --dataset_name "$dataset" --model_type "$model" --xai_name "$xai_name" &
      done
    done
    wait
echo "$dataset Operation Complete"
done
