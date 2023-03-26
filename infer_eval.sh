# bash infer_eval.sh
python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-$1 --infer_test
python coco_caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json experiments/rag_7M_gpt2/checkpoint-$1/test_preds.json