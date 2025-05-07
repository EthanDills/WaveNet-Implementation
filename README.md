This is a modification of **[Vincent Herrmann's](https://github.com/vincentherrmann/pytorch-wavenet)** implementation of the **[WaveNet Architecture](https://arxiv.org/abs/1609.03499)**. The extent of the modifications are as follows:
* Created a separate dataset in train_samples/dataset.npz, encoded from train_samples/emil-telmanyi_bwv1003.wav. This .wav file was sourced from **[This github repository](https://github.com/salu133445/bach-violin-dataset/tree/main/bach-violin/audio/emil-telmanyi)**

* In `audio_data.py`, I changed `WavenetDataset.create_dataset()` to use scipy.signal and soundfile for mu-law encoding, rather than librosa
  
* In `model_logger.py`, in `TensorBoardLogger`, I changed some of the `tensorflow.summary` objects to instead be `tensorflow.compat.v1.summary objects`. The purpose of `tensorflow.compat.v1` is to allow us to continue to use objects and methods from before tensorflow 2.0 (backwards compatibility option)

* In `wavenet_training.py`, in `WaveNetTrainer.train()`, I changed `num_workers` to 0 for `self.dataloader`. This is because I ran into pickling / serialization issues when I tried to use multiprocessing. I will mention the implications of this later in the report.

* In `wavenet_modules.py`, I changed the `ConstantPad1d` class to work with a newer version of `torch.autograd.Function`.

* Changed various objects in the model that were set to run on the CPU to instead run on GPU, specifically by setting their device to "cuda"
