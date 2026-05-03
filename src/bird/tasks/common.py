"""
Base class for all Tasks.
A Task is basically a dataset of conversations, together with some
metadata and often also evaluation criteria.
Example tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SmolTalk.
"""

import random

class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    @property
    def eval_type(self):
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError

if __name__ == "__main__":
    # very lightweight test of slicing
    from MotifRecognition import MotifRecognition

    # Initialize your custom DNA task (assumes DNA.jsonl is in the working directory)
    ds = MotifRecognition(filepath="DNA.jsonl")
    print("Length of MotifRecognition: ", len(ds))
    
    # Check if the dataset loaded successfully before trying to access index 5
    if len(ds) > 5:
        ex = ds[5]
        print("5th example: ", ex)

        # Test the slicing functionality inherited from the Task base class
        ds_sliced = MotifRecognition(filepath="DNA.jsonl", start=5, stop=10)
        print("Length of sliced MotifRecognition[5:10]: ", len(ds_sliced))
        print("0th example of sliced MotifRecognition: ", ds_sliced[0])

        print("They match: ", ex == ds_sliced[0])
    else:
        print("Dataset has fewer than 6 examples. Check your DNA.jsonl file path and contents.")