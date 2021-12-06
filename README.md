# Reassessed labels for the ILSVRC-2012 ("ImageNet") validation set

This repository contains data and example code for computing the "ReaL accuracy"
on ImageNet used in our paper [Are we done with ImageNet?](https://arxiv.org/abs/2006.07159).

## Example code for computing ReaL accuracy

The following example code is licensed under the Apache 2.0 license, see LICENSE file.
Disclaimer: This is not an officially supported Google product.

### NumPy

```python
import json
import numpy as np

real_labels = json.load('real.json')
predictions = np.argmax(get_model_logits(val_images), -1)

# Assuming val_images are ordered correctly (from ILSVRC2012_val_00000001.JPEG to ILSVRC2012_val_00050000.JPEG)
is_correct = [pred in real_labels[i] for i, pred in enumerate(predictions) if real_labels[i]]
real_accuracy = np.mean(is_correct)

# If the images were not sorted, then we need the filenames to map.
real_labels = {f'ILSVRC2012_val_{(i+i):08d}.JPEG': labels for i, labels in enumerate(json.load('real.json'))}
is_correct = [pred in real_labels[val_fnames[i]] for i, pred in enumerate(predictions) if real_labels[i]]
real_accuracy = np.mean(is_correct)
```

### PyTorch

We hope to make our labels easier to use by integrating them with [torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet) after the release.

Ross Wightman ([@rwightman](https://github.com/rwightman)) has now included [evaluation on the ReaL labels](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet-real.csv)
in his well known [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) repository.
Have a look at [real_labels.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/real_labels.py)
and the way it is used in [validate.py](https://github.com/rwightman/pytorch-image-models/blob/078a51dbac2ec4e401e166a3aec0b3c613e6c06f/validate.py#L196-L197)
for a good usage example in PyTorch.

### TensorFlow Datasets

Our labels are available in the [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_real) library.

Usage example:

```python
import tensorflow_datasets as tfds

data = tfds.builder('imagenet2012_real')
data = data.as_dataset(split='validation')
data_iter = data.as_numpy_iterator()
example = data_iter.next()
example['image'], example['original_label'], example['real_label']
```

## Description of the files

### real.json

This file is a list of 50 000 lists which contain the "Reassessed Labels" used for
evaluation in the paper.

The outer index of the list corresponds to the validation files, sorted by name.
That means, the first list holds all valid labels for the file `ILSVRC2012_val_00000001.JPEG`,
the second list holds all valid labels for the file `ILSVRC2012_val_00000002.JPEG`, etc.

Note that lists can be empty, in which case the file should not be included in
the evaluation, nor in computing mean accuracy.
These are images where the raters found none of the labels to reasonably fit.

### scores.npz

This contains the scores for each rated `(image, label)` pair, as computed by
the Dawid & Skene 1979 algorithm.
These numbers were used to draw the full precision-recall curve in Figure 3, and
could be used if you want to try different operating points than the one we used.

This is a compressed numpy archive, that can be loaded as follows:

```python
data = np.load('scores.npz')
scores, info = data['scores'], data['info']
```

Then, `scores` is an array of N floats in [0,1], and `info` is an array of N
`(fname, label)` pairs describing which validation file and label is being scored.

### raters.npz, golden.npz, raters_golden.npz

These are the raw rater votes, the "golden" labels provided by the 5 expert
raters (paper authors) and the _all_ rater's answers to those golden questions.

All files follow this format:

```python
data = np.load('scores.npz')
scores, info, ids = data['scores'], data['info'], data['ids']
```

Here, `scores` is a `RxNx3` binary tensor, where a 1 is placed when rater `R`
answered question `N` with `no/maybe/yes`, respectively.
Again, `info` is a list of `N` `(fname, label)` pairs.
Additionally, `ids` is a list of rater length `R` containing rater IDs, which
can be used to match raters across `raters.npz` and `raters_golden.npz`.

## List of ImageNet training files

We also release the [list of training file-names](https://github.com/google-research/reassessed-imagenet/releases/download/v1.0/fnames_clean.txt)
used in Section 6 of the paper.
