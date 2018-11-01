from typing import Optional, Tuple, List, Dict, Union

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.autograd import Variable

from .encoder_base import _EncoderBase
from .lstm_cell_with_projection import LstmCellWithProjection

RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
RnnStateStorage = Tuple[torch.Tensor, ...]


class ElmobiLm(_EncoderBase):

    def __init__(self, config: Dict, use_cuda: bool = False):
        super(ElmobiLm, self).__init__(stateful=True)
        self.config = config
        self.use_cuda = use_cuda
        self.input_size = config['encoder']['projection_dim']
        self.hidden_size = config['encoder']['projection_dim']
        self.cell_size = config['encoder']['dim']
        self.num_layers = config['encoder']['n_layers']
        self.memory_cell_clip_value = config['encoder']['cell_clip']
        self.state_projection_clip_value = config['encoder']['proj_clip']
        self.recurrent_dropout_probability = config['dropout']

        forward_layers, backward_layers = [], []
        for layer_index in range(self.num_layers):
            forward_layer = LstmCellWithProjection(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                cell_size=self.cell_size,
                go_forward=True,
                recurrent_dropout_probability=self.recurrent_dropout_probability,
                memory_cell_clip_value=self.memory_cell_clip_value,
                state_projection_clip_value=self.state_projection_clip_value
            )
            backward_layer = LstmCellWithProjection(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                cell_size=self.cell_size,
                go_forward=False,
                recurrent_dropout_probability=self.recurrent_dropout_probability,
                memory_cell_clip_value=self.memory_cell_clip_value,
                state_projection_clip_value=self.state_projection_clip_value
            )
            self.add_module(f'forward_layer_{layer_index}', forward_layer)
            self.add_module(f'backward_layer_{layer_index}', backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._lstm_forward, inputs, mask)

        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        # Add back invalid rows which were removed in the call to sort_and_run_forward.
        if num_valid < batch_size:
            zeros = stacked_sequence_output.data.new(
                num_layers, batch_size - num_valid, returned_timesteps, encoder_dim).fill_(0)
            zeros = Variable(zeros)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

            # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.data.new(num_layers, batch_size - num_valid, state_dim).fill_(0)
                zeros = Variable(zeros)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.data.new(
                num_layers, batch_size, sequence_length_difference, stacked_sequence_output[0].size(-1)).fill_(0)
            zeros = Variable(zeros)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)
        return stacked_sequence_output.index_select(1, restoration_indices)

    def _lstm_forward(self,
                      inputs: PackedSequence,
                      initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                      ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
          A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
          A tuple (state, memory) representing the initial hidden state and memory
          of the LSTM, with shape (num_layers, batch_size, 2 * hidden_size) and
          (num_layers, batch_size, 2 * cell_size) respectively.
        Returns
        -------
        output_sequence : ``torch.FloatTensor``
          The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
        final_states: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
          The per-layer final (state, memory) states of the LSTM, with shape
          (num_layers, batch_size, 2 * hidden_size) and  (num_layers, batch_size, 2 * cell_size)
          respectively. The last dimension is duplicated because it contains the state/memory
          for both the forward and backward layers.
        """
        if initial_state is None:
            hidden_states = [None] * len(self.forward_layers)  # type: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise Exception("Initial states were passed to forward() but the number of "
                            "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
            backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))

            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None
                backward_state = None

            forward_output_sequence, forward_state = forward_layer(
                forward_output_sequence, batch_lengths, forward_state)
            backward_output_sequence, backward_state = backward_layer(
                backward_output_sequence, batch_lengths, backward_state)

            # Skip connections, just adding the input to the output.
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(torch.cat([forward_output_sequence, backward_output_sequence], -1))
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                                 torch.cat([forward_state[1], backward_state[1]], -1)))

        stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of shape
        # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
        # respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple = (torch.cat(final_hidden_states, 0), torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple
