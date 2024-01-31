import torch
import torch.nn.functional as F

class RandomSegmentMask(torch.nn.Module):
    
    def __init__(self, start_probability=0.05, markov_probability=0.7):
        '''
        mask_probability : Probability that determines if a 
            channel will be masked
        start_probability : Probability that determines if a 
            timestep is the beginning of a masked segment
        markov_probability : Probability that determines the
            length of masked segments
        '''
        super().__init__()
        #self.mask_prob = mask_probability
        self.start_prob = start_probability
        self.markov_prob = markov_probability
        # self.not_active = (
        #     self.mask_prob * self.start_prob * self.markov_prob
        # ) == 0
        self.not_active = (self.start_prob * self.markov_prob) == 0
        
    def forward(self, x, mask_value=0):
        if self.not_active or not self.training:
            return x, torch.zeros_like(x)
        n_batch, n_seqs, seq_len = x.shape
        all_filters = torch.tril(torch.ones([seq_len, seq_len]))
        all_filters = torch.flip(all_filters, (-1,))
        window_length_probs = self.markov_prob**torch.arange(0, seq_len)[:-1]
        
        #mask_seq = torch.rand(size=[n_batch, n_seqs, 1]) < self.mask_prob
        idx_mask = torch.rand(size=[n_batch, n_seqs, seq_len]) < self.start_prob
        #b_idx, s_idx, t_idx = torch.where(torch.logical_and(mask_seq, idx_mask))
        b_idx, s_idx, t_idx = torch.where(idx_mask)

        mask_size = torch.distributions.Categorical(window_length_probs).sample((n_seqs,))#[s_idx]
        filters = all_filters[mask_size]
        
        mask = torch.zeros([n_batch, n_seqs, seq_len])
        mask[b_idx, s_idx, t_idx] = 1.
        
        mask = F.conv1d(mask, filters[:,None], padding=seq_len-1, groups=n_seqs) >= 1
        mask = mask[...,:seq_len]
        #print(mask.sum(), mask_size)
        
        # debug
        if False:
            # Visualize with:
            # seq = torch.ones([1, 10, 56])
            # rsm = RandomSegmentMask(0.5, 0.05, 0.98)
            # mask, (b_idx, s_idx, t_idx, sizes) = rsm(seq)
            # plt.imshow(mask[0])
            # plt.scatter(t_idx, s_idx, s=8)
            # plt.scatter(np.minimum(seq_len-1, t_idx + sizes), s_idx, s=8, c='r')
            return mask, (b_idx, s_idx, t_idx, mask_size[s_idx])
        
        x[mask] = mask_value
        
        return x, mask