# for example bash infer_eval.sh 10000
set -e
export TOKENIZERS_PARALLELISM=false
echo "Inference now"
python infer.py --model_path experiments/$1 --checkpoint_path checkpoint-$2 --infer_test --features_path features/coco_cdn.hdf5
echo "Evaluation now"
python coco_caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json experiments/$1/checkpoint-$2/test_preds.json
