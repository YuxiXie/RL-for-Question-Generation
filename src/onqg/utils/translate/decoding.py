import numpy as np


def get_decoded_seqs(output_seqs, word2index, max_len, batch_size):
    decoded_seqs = []
    decoded_lens = []
    for b in range(batch_size):
        decoded_seq = []
        decoded_seq_mask = [0]*max_len
        for t in range(max_len):
            idx = int(output_seqs[t][b])
            if idx == word2index[EOS_token]:
                decoded_seq.append(idx)
                break
            else:
                decoded_seq.append(idx)
                decoded_seq_mask[t] = 1
        decoded_lens.append(len(decoded_seq))
        decoded_seq += [word2index[PAD_token]]*(max_len - len(decoded_seq))
        decoded_seqs.append(decoded_seq)

    decoded_lens = np.array(decoded_lens)
    decoded_seqs = np.array(decoded_seqs)
    return decoded_seqs, decoded_lens