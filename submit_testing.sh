#!/bin/sh

# Don't need to initialize / activate Colab's Conda 

# Resolve script directory
SCRIPT_PATH="$(readlink -f "$0")"

# main directory where the folders are
main_directory="$(dirname "$SCRIPT_PATH")"
echo "Script directory (relative): $main_directory"

# input, output, and TredNet folders
source_input=${main_directory}/input_training_trednet/testing
source_output=${main_directory}/model_output/testing
source_work=${main_directory}
source_fasta=${main_directory}/fasta/hg38.fa
source_model_phase_I=${main_directory}/v0.1

echo $source_input
# list of biosamples to use
biosample_file=${main_directory}/list_of_biosamples_testing.txt

while IFS= read -r line; do
	echo $line

       # create a biosample folder for the model
       dl_dir="$source_output/${line}.phase_two.hg38"
       mkdir -p $dl_dir
       cd $dl_dir/

       # copy biosample model and change folders name in the two phase py file
       cp $source_work/back_run_two_phase.py run_two_phase.${line}.py
       sed -i -e "s@sourceinput@$source_input@g" run_two_phase.${line}.py
       sed -i -e "s@dl_dir@$dl_dir@g" run_two_phase.${line}.py
       sed -i "s/EID/$line/g" run_two_phase.${line}.py
       sed -i -e "s@sourcefasta@$source_fasta@g" run_two_phase.${line}.py
       sed -i -e "s@sourcemodelI@$source_model_phase_I@g" run_two_phase.${line}.py
       cd $source_work/

#	python ${dl_dir}/run_two_phase.${line}.py
  conda run -n tf2.15-gpu python ${dl_dir}/run_two_phase.${line}.py

       cd $source_work/

done < $biosample_file
