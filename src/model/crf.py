"""
Based on https://github.com/thespectrewithin/joint_align/blob/master/crf.py with better
support of masking, allowing masking in the middle of a sentence.
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def logsumexp(x, dim=None):
    """
    Args:
        x: A pytorch tensor (any dimension will do)
        dim: int or None, over which to perform the summation. `None`, the
             default, performs over all axes.
    Returns: The result of the log(sum(exp(...))) operation.
    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + torch.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))


class ChainCRF(nn.Module):
    def __init__(self, input_size, num_labels, bigram=True):
        """
        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
        """
        super(ChainCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram

        # state weight tensor
        self.state_nn = nn.Linear(input_size, self.num_labels)
        if bigram:
            # transition weight tensor
            self.trans_nn = nn.Linear(input_size, self.num_labels * self.num_labels)
            self.register_parameter("trans_matrix", None)
        else:
            self.trans_nn = None
            self.trans_matrix = Parameter(
                torch.Tensor(self.num_labels, self.num_labels)
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.state_nn.bias, 0.0)
        if self.bigram:
            nn.init.xavier_uniform_(self.trans_nn.weight)
            nn.init.constant_(self.trans_nn.bias, 0.0)
        else:
            nn.init.normal_(self.trans_matrix)

    def forward(self, input, mask):
        """
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor
                the mask tensor with shape = [batch, length]

        Returns: Tensor
            the energy tensor with shape = [batch, length, num_label, num_label]
        """
        batch, length, _ = input.size()

        # compute out_s by tensor dot [batch, length, input_size] * [input_size,
        # num_label] thus out_s should be [batch, length, num_label] --> [batch, length,
        # num_label, 1]
        out_s = self.state_nn(input).unsqueeze(2)

        if self.bigram:
            # compute out_s by tensor dot: [batch, length, input_size] * [input_size,
            # num_label * num_label] the output should be [batch, length, num_label,
            # num_label]
            out_t = self.trans_nn(input).view(
                batch, length, self.num_labels, self.num_labels
            )
            output = out_t + out_s
        else:
            # [batch, length, num_label, num_label]
            output = self.trans_matrix + out_s

        output = output * mask.unsqueeze(2).unsqueeze(3)

        return output

    def loss(self, energy, target, mask):
        """
        Args:
            energy: Tensor
                the energy tensor with shape = [batch, length, num_label, num_label]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor
                the mask tensor with shape = [batch, length]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        """
        batch, length = target.size()
        # shape = [length, batch, num_label, num_label]
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch]
        target_transpose = target.transpose(0, 1)
        # shape = [length, batch, 1]
        mask_transpose = mask.unsqueeze(2).transpose(0, 1)

        # shape = [batch, num_label]
        partition = None

        # shape = [batch]
        batch_index = torch.arange(0, batch, dtype=torch.long, device=target.device)
        prev_label = torch.zeros(batch, dtype=torch.long, device=target.device)
        prev_label = prev_label.fill_(self.num_labels - 1)
        tgt_energy = torch.zeros(batch, device=target.device)

        for t in range(length):
            # shape = [batch, num_label, num_label]
            curr_energy = energy_transpose[t]
            mask_t = mask_transpose[t]
            if t == 0:
                partition = curr_energy[:, -1, :]
            else:
                # shape = [batch, num_label]
                partition_new = logsumexp(curr_energy + partition.unsqueeze(2), dim=1)
                partition = partition + (partition_new - partition) * mask_t
            tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t].data]
            prev_label_new = target_transpose[t].data
            prev_label = (
                prev_label + (prev_label_new - prev_label) * mask_t.squeeze(1).long()
            )

        return (logsumexp(partition, dim=1) - tgt_energy).mean()

    def decode(self, energy, mask):
        """
        Args:
            energy: Tensor
                the energy tensor with shape = [batch, length, num_label, num_label]
            mask: Tensor
                the mask tensor with shape = [batch, length]

        Returns: Tensor
            decoding results in shape [batch, length]
        """
        # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
        # For convenience, we need to dimshuffle to (n_time_steps, n_batch, num_labels,
        # num_labels)
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch, 1]
        mask_transpose = mask.unsqueeze(2).transpose(0, 1).long()

        # the last row and column is the tag for pad symbol. reduce these two dimensions
        # by 1 to remove that.
        # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t =
        # num_labels - 1.
        energy_transpose = energy_transpose[:, :, :-1, :-1]

        length, batch_size, num_label, _ = energy_transpose.size()

        batch_index = torch.arange(
            0, batch_size, dtype=torch.long, device=energy.device
        )
        pi = torch.zeros([length, batch_size, num_label, 1], device=energy.device)
        pointer = torch.zeros(
            [length, batch_size, num_label], dtype=torch.long, device=energy.device
        )
        dummy_pointer = torch.arange(self.num_labels - 1, device=energy.device)
        back_pointer = torch.zeros(
            [length, batch_size], dtype=torch.long, device=energy.device
        )

        # [length, batch_size, num_label, 1]
        pi[0] = energy[:, 0, -1, :-1].unsqueeze(2)
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1]
            mask_t = mask_transpose[t]
            pi_t, pointer_t = torch.max(energy_transpose[t] + pi_prev, dim=1)
            pointer[t] = pointer_t * mask_t + dummy_pointer * (1 - mask_t)
            pi[t] = (pi_t * mask_t).unsqueeze(2) + pi[t - 1] * (1 - mask_t.unsqueeze(2))

        _, back_pointer[-1] = torch.max(pi[-1].squeeze(2), dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]

        return back_pointer.transpose(0, 1)
