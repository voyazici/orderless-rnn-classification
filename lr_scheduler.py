from torch.optim.lr_scheduler import _LRScheduler

class CyclicalLR(_LRScheduler):
    """Cyclical learning through the whole training."""
    def __init__(self, optimizer, lr_high, lr_low, cycle_length=3):
        """Init function."""
        self.current_lr = [lr_high]
        self.lr_high = lr_high
        self.lr_low = lr_low
        self.cycle_length = cycle_length
        self.curr_iter = 0
        super(CyclicalLR, self).__init__(optimizer)

    def print_lr(self):
        """Return lr value without incrementing current iteration."""
        return self.current_lr

    def reset_curr_iter(self):
        """Reset the current iteration value."""
        self.curr_iter = 0

    def set_curr_iter(self, iterations):
        """Set the current iteration value."""
        self.curr_iter = iterations

    def get_lr(self):
        """Get the learning rates for the current iteration."""
        inv_cl = 1.0 / self.cycle_length
        cycle_step = self.curr_iter % (self.cycle_length + 1)
        t_i = inv_cl * cycle_step
        self.current_lr = [t_i * self.lr_low + (1 - t_i) * self.lr_high]
        self.curr_iter += 1
        return self.current_lr
