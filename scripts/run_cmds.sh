#/bin/bash
CKPT_DIR=checkpoints/test/
MAIN_SCRIPT_PATH=../run.py
CONFIG_FILE=test_config.json

run_test () {
  echo "Testing $1"
  python $MAIN_SCRIPT_PATH -c $CONFIG_FILE ${@:1:99} >test_${1}.log 2>error_${1}.log && echo "$1 succeeded" || { cat test_${1}.log && cat error_${1}.log && echo "$1 failed" && exit 1; }
}

python $MAIN_SCRIPT_PATH self_play --help

run_test self_play

run_test cross_play -d $CKPT_DIR

run_test clash --no-render $CKPT_DIR`ls $CKPT_DIR | cut -f 1 | head -n1` $CKPT_DIR`ls $CKPT_DIR | cut -f 1 | tail -n1`

run_test train -ckpt $CKPT_DIR`ls $CKPT_DIR | cut -f 1 | tail -n1`

CONFIG_FILE=test_config_for_hopt.json
run_test hopt -n 20

rm -rf checkpoints
