from dataclasses import dataclass


@dataclass
class Procedure:
    """The base definition of a procedure as used across all datasets."""

    _input: str
    output: str
    steps: list[str]

    def to_json(self):
        return self.__dict__

    def _set_input(self, input_str):
        self.input = input_str

    def _set_output(self, output_str):
        self.output = output_str

    def _set_steps(self, steps):
        self.steps = steps

    def _add_step(self, procedure_step, idx=None):
        if idx is None:
            self.steps.append(procedure_step)
        else:
            self.steps[idx] = procedure_step

    def _get_num_steps_procedure(self):
        return len(self.steps)
