torch-model-archiver -f --model-name dog_breed_classification --version 1.0 --model-file dog_breed_classification_arch.py --serialized-file dog_breed_classification.pth --handler image_classifier --extra-files index_to_name.json --export-path model_store

torchserve --start --model-store model_store/ --workflow-store wf_store/ --ncs

curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=dog_breed_classification.mar"

curl http://127.0.0.1:8080/predictions/dog_breed_classification -T Dog.jpg
{
  "Petit_basset_griffon_vendeen": 0.9298110008239746,
  "French_bulldog": 0.046617258340120316,
  "American_foxhound": 0.017280232161283493,
  "Australian_terrier": 0.002052989089861512,
  "Doberman_pinscher": 0.0009731495520099998
}
