`run_elpis.py` is an adaptation of `run_common_voice.py` to handle data
that has been preprocessed by Elpis.

`run_common_voice.py` is a script that is used to prepare and train wav2vec2
models using data from the Common Voice corpora. Information on how to call
that script can be found in README.md.

[wav2vec2]
(https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf). The architecture is not totally dissimilar
to what you might find in a package like ESPnet, or what was used in
persephone. Unfortunately the wav2vec2 paper isn't super readable because it
doesn't really stand alone that well from the wav2vec 1 paper and the other
stuff it builds on. But the crucial difference is that it is designed so that you can
pre-train it unsupervisedly on lots of audio. This is beneficial in the
low-resource context because:
- Far more untranscribed audio typically exists than transcribed data, because
  of the transcription bottleneck.
- But even more crucically, you don't really need to run unsupervised
  pre-training in the target language. You can just use a model that has
  already been pre-trained multilingually on thousands of hours of speech. The
  model used by this script was trained on data in 53 languages. It was
  unsupervised, so it learns it's own notion of discrete phonetic/phonemic
  units and isn't wedded to the orthography of the languages it was pretrained
  on.
This pre-training means the model has a good sense of phonetics/phonology and
is robust to channel variability. With data in a new target language, all it
has to do is learn a correspondence between it's latent states and the target
orthography.

`run_elpis.py` code largely overlaps with that script and has similar calling
convention, but some differences are as follows.

Added arguments in calling the script:
* `elpis_data_dir` the data containing pre-processed speech/transcriptions from
  elpis.
* `train_size` the fraction of data used for training, with the rest being used
  for dev and test sets. Default is 0.8 but this can become smaller with more
  training data.
* `split_seed` The seed used to create train/dev/test splits. You can just
  leave this as the default if you want.

Description of added functionality:
- The elpis data is split into train/dev/test splits. Common voice data already
  has this done, so this functionality needed to be added.
- A hugging Face style 'dataset' object is created from the Elpis JSON.
- The audio files referenced are then resampled to 16kHz.
- Utterance WAV data is extracted from teh full audio using the `start_ms` and
  `stop_ms` info from the Elpis JSON.
- Output hypotheses and references are written to the experiment directory at
  each eval step with a timestamp, so the practictioner can see how the quality
  of the predictions are improving.
- There's also this ElpisTokenizer object that Ben G has documented in the code
  better, to handle multi-character phonemic units.

