input_path='./data/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json'   # CMB-Exam
# input_path='./data/CMB-Clin/CMB-Clin-qa.json'                             # CMB-Clin

LOCAL_RANK=0

task_name='Exam' 

model_id="chatglm3_6b" # which model to evaluate
mkdir -p logs/${task_name}/

# accelerate launch --main_process_port 27274 --config_file ./configs/accelerate_config.yaml \
# torchrun --nnodes=1 --nproc_per_node=2 \
python \
./src/generate_answers.py --use_cot \
--model_id=$model_id \
--all_gather_freq=20 \
--input_path=$input_path \
--output_path=./result/${task_name}/${model_id}/modelans.json \
--batch_size 1 \
--model_config_path="./configs/model_config.yaml" 