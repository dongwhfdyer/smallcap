# for example bash infer_eval.sh 10000
set -e
export TOKENIZERS_PARALLELISM=false
echo "Inference now"
python infer.py --model_path experiments/exp_0406-2039 --checkpoint_path checkpoint-$1 --infer_test --features_path features/coco_cdn.hdf5
echo "Evaluation now"
python coco_caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json experiments/exp_0406-2039/checkpoint-$1/test_preds.json
