import torch
import torch.nn as nn
from torchaudio.functional import pitch_shift


class RandomRepeat(nn.Module):
    """
    Repeats a short audio waveform a random number of times with random offsets
    to fill a fixed-length output.

    This is useful for augmenting short, distinct sound events (e.g., a "bark" or a
    "siren") by placing them multiple times within a fixed-length background.

    Args:
        sample_rate (int): The sample rate of the input waveform.
        target_duration_s (float): The desired duration of the output waveform in seconds.
                                   Defaults to 4.0.
    """

    def __init__(self, sample_rate: int, target_duration_s: float = 4.0, random_state=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_duration_s = target_duration_s
        self.target_length = int(self.target_duration_s * self.sample_rate)
        self.random_state = random_state

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Processes a waveform to repeat it randomly within a fixed-length bed.

        Args:
            waveform (torch.Tensor): The input waveform tensor, expected shape (1, T).

        Returns:
            torch.Tensor: The augmented waveform of shape (1, target_length),
                          containing random repetitions of the input.
        """
        assert waveform.dim() == 2, \
            f"Input must be a single sample with shape (C, T), but got a tensor with {waveform.dim()} dimensions."

        if self.random_state is not None:
            generator = torch.Generator().manual_seed(self.random_state)
        else:
            generator = torch.Generator()

        waveform_length = waveform.shape[-1]

        # create the fixed-length empty "bed"
        augmented_waveform = torch.zeros(waveform.shape[0], self.target_length, dtype=waveform.dtype)
        if waveform_length == 0:
            return augmented_waveform  # handle empty input case

        # calculate the maximum number of times the waveform can fit
        max_possible_cycles = self.target_length // waveform_length
        if max_possible_cycles == 0:
            # the waveform is longer than the target duration - just crop a desired-len-section from it.
            start_index = torch.randint(low=0, high=waveform_length - self.target_length + 1, size=(1,),
                                        generator=generator).item()
            return waveform[..., start_index : start_index+self.target_length]

        # choose randomly how many times we wanna cycle
        num_cycles_to_place = torch.randint(low=1, high=max_possible_cycles + 1, size=(1,),
                                            generator=generator).item()

        current_position = 0
        for _ in range(num_cycles_to_place):
            remaining_length = self.target_length - current_position
            if remaining_length < waveform_length:
                break  # not enough space to place another full waveform

            # calculate the maximum possible offset for this cycle
            # and put new wav instance with random offset from possible ones.
            max_offset = remaining_length - waveform_length

            offset = torch.randint(low=0, high=max_offset + 1, size=(1,),
                                   generator=generator).item()

            start_index = current_position + offset
            end_index = start_index + waveform_length
            augmented_waveform[..., start_index:end_index] = waveform

            current_position = end_index

        return augmented_waveform


class RandomizedPitchShift(nn.Module):
    """
    A randomized pitch shift transform for audio data augmentation.

    This class composes `torchaudio.transforms.PitchShift` and adds a
    layer of randomness, choosing a pitch shift amount uniformly from a
    specified range. This helps the model generalize to variations in
    pitch, such as different speakers or instruments.

    Args:
        sample_rate (int): The sample rate of the input waveform in Hertz.
        shift_range (int): The maximum number of semitones to shift, both up and down.
                           A random integer value will be chosen from
                           [-shift_range, +shift_range]. Default is 3.
        **kwargs: Additional keyword arguments to pass to the internal
                  `torchaudio.transforms.PitchShift` constructor.
    """

    def __init__(self, sample_rate: int, n_fft: int, shift_range: int = 2, random_state=None, **kwargs):
        super().__init__()
        if not isinstance(shift_range, int) or shift_range < 0:
            raise ValueError("`shift_range` must be a non-negative integer.")

        # We compose the PitchShift transform, but don't set n_steps here.
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.shift_range = shift_range
        self.random_state = random_state

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applies a randomized pitch shift to the input waveform.

        Args:
            waveform (torch.Tensor): The input waveform tensor. Should be of
                                     shape `(..., time)`.

        Returns:
            torch.Tensor: The pitch-shifted waveform, with the same shape
                          as the input.
        """
        if self.random_state is not None:
            generator = torch.Generator().manual_seed(self.random_state)
        else:
            generator = torch.Generator()
        # generate a random integer from the range [-shift_range, shift_range].
        n_steps = torch.randint(low=-self.shift_range, high=self.shift_range + 1, size=(1,),
                                generator=generator).item()

        # Apply the pitch shift to the waveform.
        return pitch_shift(waveform=waveform, sample_rate=self.sample_rate, n_fft=self.n_fft, n_steps=n_steps)

class UnsqueezeBatch(nn.Module):
    """Add a fake batch dimension to [C, T] → [1, C, T]"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C, T]
        if x.dim() != 2:
            raise ValueError(f"Expected input [C, T], got {x.shape}")
        return x.unsqueeze(0)  # [1, C, T]

class SqueezeBatch(nn.Module):
    """Remove the fake batch dimension [1, C, T] → [C, T]"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected input [1, C, T], got {x.shape}")
        return x.squeeze(0)  # [C, T]