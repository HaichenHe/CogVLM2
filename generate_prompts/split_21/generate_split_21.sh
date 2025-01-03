python -W ignore -u generate_prompts/tools/run_generate.py \
  --init_method tcp://localhost:10019 \
  --cfg generate_prompts/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
  --opts DATA.PATH_TO_DATA_DIR /opt/data/private/hhc/workdir/CogVLM2/generate_prompts/split_21 \
  OUTPUT_GENERATE_PROMPT /opt/data/private/hhc/workdir/CogVLM2/generate_prompts/split_21/split_21_prompts.json \
  TRAIN_FULL_FILE /opt/data/private/hhc/workdir/CogVLM2/generate_prompts/train_full.csv \
  DATA.PATH_PREFIX /opt/data/private/hhc/recognition/Datasets/kinetics400 \
  DATA.PATH_LABEL_SEPARATOR , \
  TRAIN.ENABLE True \
  TRAIN.BATCH_SIZE 1 \
  NUM_GPUS 1 \
  DATA.DECODING_BACKEND "pyav" \
  AUG.ENABLE False \
  AUG.NUM_SAMPLE 1 \
  