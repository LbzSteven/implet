#!/bin/bash
MODELS=("FCN" "InceptionTime")
for model in "${MODELS[@]}"; do

  SOURCE_PATH="../../PretrainModels/TimeSeriesClassifications/$model"  # A 文件夹路径
  DEST_PATH="models/$model"
  FOLDERS=("Computers" "Chinatown" "ECG200" "DistalPhalanxOutlineCorrect" "Yoga" "GunPoint" "PowerCons" "Earthquakes" "FordA" "Strawberry" "Wafer" "HandOutlines")

  for folder in "${FOLDERS[@]}"; do
      if [ -d "$SOURCE_PATH/$folder" ]; then
          cp -r "$SOURCE_PATH/$folder" "$DEST_PATH"
          echo "move: $folder to $DEST_PATH"
      else
          echo "folder $folder doesn't exist in $SOURCE_PATH"
      fi
  done
done
echo "Operation Complete"