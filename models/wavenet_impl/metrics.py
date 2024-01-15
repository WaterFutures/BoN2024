from torchmetrics import (
    MeanMetric, MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
)

class MulticlassMetrics:

    def __init__(self, num_classes, prefix, device):
        self.prefix = prefix
        self.metrics = {
            'mse' : MeanSquaredError().to(device),
            'mae' : MeanAbsoluteError().to(device),
            'mape' : MeanAbsolutePercentageError().to(device),
        }
        # apply prefix
        self.metrics = { f'{self.prefix}{k}' : fn for k, fn in self.metrics.items() }
        self.mean_grad_norm = None
        self.mean_loss = None
        self.grad_norm_key = f'{self.prefix}gradient_norm'
        self.loss_key = f'{self.prefix}loss'
        self.device = device

    def result_and_reset(self):
        means = {}
        for k in self.metrics:
            means[k] = self.metrics[k].compute().cpu()
            self.metrics[k].reset()
        if self.mean_grad_norm is not None:
            means[self.grad_norm_key] = self.mean_grad_norm.compute().cpu()
            self.mean_grad_norm.reset()
        if self.mean_loss is not None:
            means[self.loss_key] = self.mean_loss.compute().cpu()
            self.mean_loss.reset()
        return means

    def compute_metrics(self, output, ground_truth, loss=None, gradient_norm=None):
        results = {}
        for k in self.metrics:
            m = self.metrics[k](output, ground_truth)
            results[k] = m
        if gradient_norm is not None:
            if self.mean_grad_norm is None:
                self.mean_grad_norm = MeanMetric().to(self.device)
            self.mean_grad_norm.update(gradient_norm)
            results[self.grad_norm_key] = gradient_norm
        if loss is not None:
            if self.mean_loss is None:
                self.mean_loss = MeanMetric().to(self.device)
            self.mean_loss.update(loss)
            results[self.loss_key] = loss
        return results
