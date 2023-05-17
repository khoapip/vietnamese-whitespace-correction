# whitespace-removal

## Requirements

```
pip install -r requirements.txt
```

## Generate Dataset
```
./generate_data.sh
```
Automatically download wikilingua dataset. Split into sentences and generate input, labels. See ```process``` in ```generate_dataset.py``` for more information.

Todo cases:
* [ ] Add harder cases to process function (insert corpus with whitespaces in between for single word whitespace removal)
* [ ] Inference code and trained models
* [ ] Test with PhoBART
* [ ] Perform a more thorough research 
* [ ] Demo with streamlit



