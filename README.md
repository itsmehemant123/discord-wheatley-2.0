# discord-wheatley-2.0

### Setup

- Install dependencies with:

```bash
pip install -r requirements.txt
```

- Install tensorflow (GPU version is recommended for training).

- Create `auth.json`, and place it inside the `config` folder. Its content should be:

```json
{
   "token": "<your_token>",
   "client_id": "<your_client_id>"
}
```

Where `<your_token>` and `<your_client_id>` are fetched from discord.

### Pre-processing

- Download the corpus, and create the `.from` and `.to` file pairs according to the standard nmt architecture.

- Add the dev and test pairs of `.from` and `.to` with small subset of data.

- Create the vocabulary file with the top X words from the dataset (Anything above 5000).

- Place all the above files in a `dataset` folder. This folder path will be passed onto the script while training and inference.

### Training

Train the model by:
```sh
python -m nmt.nmt --src=from --tgt=to 
--vocab_prefix=<path_of_dataset_folder>/vocab 
--train_prefix=<path_of_dataset_folder>/train 
--dev_prefix=<path_of_dataset_folder>/dev 
--test_prefix=<path_of_dataset_folder>/test 
--out_dir=<path_of_output_folder> 
--num_train_steps=60000 --steps_per_stats=100 
--num_layers=2 --num_units=128 --dropout=0.2 
--metrics=bleu
```

Where, `<path_of_dataset_folder>` contains the 
- `vocab.from` and `vocab.to` containing the corresponding top X used words. 
- `train.from` and `train.to` containing the training data.
- `dev.from` and `dev.to` containing the dev/validation data.
- `test.from` and `test.to` containing the test data.

The `<path_of_output_folder>` is an empty folder where the trained model and the checkpoints are saved.

Try adding `beam_search`, `attention` and switch the `encoder` and `decoder` configuration parameters and test performance.

### How to run

- Once you have a desired checkpoint, add the path to the folder in `Line 22` in `wheatley.py` (`<path_to_model>`).

- Run the script with:

```bash
python client.py
```

