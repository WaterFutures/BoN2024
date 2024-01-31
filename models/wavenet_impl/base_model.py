import torch
from tqdm.auto import tqdm
from .random_mask import RandomSegmentMask
from .processors import Preprocessor
from .torch_dataset import load_dataframe
import logging
from .metrics import MulticlassMetrics, MeanMetric
from torchmetrics.wrappers import MultioutputWrapper
import numpy as np
from pprint import pprint

CHANNEL_DIM = -2

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, device=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_metric = torch.tensor(1e10, dtype=torch.float32).to(device)

    def __call__(self, metric):
        if (metric + self.min_delta) < self.min_metric:
            self.min_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print('Stopped Early:', metric.item(), self.min_metric.item())
                return True
        return False


class BaseModel(torch.nn.Module):
    
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.training_epoch = torch.nn.Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        self.global_step = torch.nn.Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        self.normalizer = None
        self.logger = logging.getLogger()
        
    def save(self, ckpt_path, optimizer=None):
        get_opt_state = getattr(optimizer, 'state_dict', None)
        checkpoint = {
            'epoch' : self.training_epoch.item(),
            'global_step' : self.global_step.item(),
            'model_state_dict' : self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = get_opt_state()
        
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def load(self, ckpt_path, optimizer=None):
        self.logger.info('Restoring model from checkpoint', ckpt_path)
        checkpoint = torch.load(ckpt_path)
        self.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return self, optimizer
    
    def fit_normalizer(self, all_training_data, config):
        # create Preprocessor
        self.normalizer = Preprocessor(config)
        # fit preprocessor on training set
        self.normalizer.fit(all_training_data)
        self.normalizer.to(config['device'])

    def normalize(self, x):
        assert self.normalizer is not None, \
        'Normalize cannot be called. Call `fit_normalizer` on the training data first.'
        return self.normalizer.transform(x)

    def unnormalize(self, x):
        assert self.normalizer is not None, \
        '`unnormalize` cannot be called. Call `fit_normalizer` on the training data first.'
        return self.normalizer.inv(x)
    
    def forward(self, *args, **kwargs):
        NotImplementedError('Overwrite this function.')

    def prepare_inputs(self, data):
        data_x = self.normalize(data['x'])
        # condition the model on the sequence
        x = data_x[..., :self.seed_len]
        # .. and additional features
        known_features = data['known_features']#[..., :self.seed_len]
        x_masked, mask = self.random_segment_masking(x)
        # predict the trailing tokens
        target = data_x[..., self.seed_len:].contiguous()
        target_mask = data['valid_mask'][..., self.seed_len:]
        residual = data.get('residuals')
        if residual is not None:
            residual = self.normalize(residual)
        return {
            'x' : x_masked, 
            'known_features' : known_features, 
            'target' : target, 
            'target_mask' : target_mask,
            'residual' : residual
        }

    def train_epoch(self, dataset, metrics, optimizer, loss_fn, clip_grads):
        for data in dataset:
            inputs = self.prepare_inputs(data)
            outputs, target, loss, grad_norm, target_mask = self.train_step(
                **inputs, loss_fn=loss_fn, optimizer=optimizer, clip_grads=clip_grads
            )
            metrics.compute_metrics(outputs, target, loss, grad_norm)

        self.training_epoch += 1

    def evaluate_epoch(self, dataset, metrics=None, loss_fn=None):
        for data in dataset:
            inputs = self.prepare_inputs(data)
            outputs, target, loss, target_mask = self.evaluation_step(**inputs, loss_fn=loss_fn)
            if metrics is not None:
                metrics.compute_metrics(outputs, target, loss)
        
    @torch.no_grad()
    def predict(self, dataset, n_steps):
        self.eval()
        predictions = []

        for data in dataset:
            inputs = self.prepare_inputs(data)
            pred, _, _ = self.predict_steps(inputs['x'], inputs['known_features'], steps=n_steps, residual=inputs['residual'])
            predictions.append(self.unnormalize(pred))

        return predictions

    def fit(self, demand_train, weather_train):
        config = self.config
        if config.get('ckpt_path') is not None:
            self.load(config.get('ckpt_path'))
            return
        
        ds = load_dataframe(demand_train, weather_train, 1, config, shuffle=True)
        self.week_avg = ds.dataset.compute_week_avg()
        ds.dataset.set_week_avg(self.week_avg)
        optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config['lr_decay'],
        )
        loss_fn = torch.nn.functional.l1_loss
        ds.dataset.to(config['device'])
        self.fit_normalizer(ds.dataset.demands, config)
        
        metrics_factory = lambda prefix=None: MulticlassMetrics(
            config['out_channels'], prefix=prefix, device=config['device']
        )
        epochs = config['epochs']
        val_every = config['val_every']
        clip_grads = config.get('grad_clipping', 1e10)
        train_metrics = metrics_factory(prefix='train/')
        start_epoch = self.training_epoch.item()
        epoch_ckpt_dir = None
        
        #stopper = EarlyStopper(patience=5, min_delta=0.02, device=config['device'])
        #stopper = EarlyStopper(patience=4, min_delta=0.05, device=config['device'])
        epoch_iterator = tqdm(range(epochs), desc=f'Training for {epochs} epochs', ncols=80, dynamic_ncols=True)
        #l = (len(ds.dataset)-1952)/12704 #* 1.5
        #epochs = int(np.minimum(10 + l * 70, 70))
        for e in epoch_iterator:
            self.train_epoch(ds, train_metrics, optimizer, loss_fn, clip_grads)
            results = train_metrics.result_and_reset()
            epoch_iterator.set_description(f'MAE: {results["train/mae"]:.4}')
            #if stopper(results['train/mae']):
            #    break
        self.adjust_bias(ds)
            
    @torch.no_grad    
    def adjust_bias(self, dataset):
        self.eval()
        if not hasattr(self, 'model_bias'):
            self.model_bias = torch.nn.Parameter(torch.zeros(1, self.out_dim, 1, dtype=torch.float32), requires_grad=False).to(self.config['device'])
            
        mean_bias = MultioutputWrapper(MeanMetric(), self.out_dim).to(self.config['device'])
        
        for data in dataset:
            inputs = self.prepare_inputs(data)
            pred, _, _ = self.predict_steps(inputs['x'], inputs['known_features'], steps=1, residual=inputs['residual'])
            target_mask = inputs['target_mask'].float()
            mean_bias((inputs['target'] - pred).squeeze(-1), weight=target_mask.squeeze(-1))
        
        self.model_bias += mean_bias.compute().unsqueeze(0).unsqueeze(-1)
        print(self.model_bias.squeeze())
    
    def forecast(self, demand_data, weather_data):
        assert (demand_data.index == weather_data.index).all()
        N_STEPS = 7*24
        config = self.config
        seq_len = config['seq_seed_len'] + N_STEPS
        demand_data = demand_data.iloc[-seq_len:]
        weather_data = weather_data.iloc[-seq_len:]
        ds = load_dataframe(demand_data, weather_data, N_STEPS, config, shuffle=True)
        ds.dataset.set_week_avg(self.week_avg)
        predictions = self.predict(ds, n_steps=N_STEPS)
        predictions = torch.cat(predictions) + self.model_bias
        return predictions.squeeze().T.cpu().numpy()
    
class AutoregressiveBaseModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.predict_week_residual = config.get('predict_week_residual')
        self.seed_len = config['seq_seed_len']
        # Currently: 1-step predictions only
        self.random_segment_masking = RandomSegmentMask(
            start_probability=config['start_probability'],
            markov_probability=config['markov_probability'],
        )
        self.condition_on_future = config.get('condition_on_future', False)

    def train_step(self, x, known_features, target, target_mask, optimizer, loss_fn, residual=None, clip_grads=1e10):
        '''
        x - node features
        edge_index - graph edge index
        edge_features - optional edge features
        mask - node-level mask for semi-supervised training
        target - ground truth
        '''
        self.train()
        target_len = target.shape[-1]
        optimizer.zero_grad()

        pred, mean, std = self.predict_next(x, known_features, residual=residual)
        # cut away seed sequence predictions
        pred = pred[..., -target_len:].contiguous()
        pred = self.unnormalize(pred)[target_mask]
        target = self.unnormalize(target)[target_mask]
        
        if self.is_probabilistic:
            mean = mean[..., -target_len:].contiguous()
            std = std[..., -target_len:].contiguous()
            mean = self.unnormalize(mean)[target_mask]
            std = std[target_mask]
            loss = loss_fn(mean, target, std**2, pred)
        else:
            loss = loss_fn(pred, target)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grads)
        optimizer.step()
        self.global_step += 1

        return pred, target, loss, grad_norm, target_mask

    @torch.no_grad()
    def evaluation_step(self, x, known_features, target=None, target_mask=None, loss_fn=None, residual=None):
        self.eval()
        loss = None
        pred, mean, std = self.predict_steps(x, known_features, steps=target.shape[-1], residual=residual)

        pred = self.unnormalize(pred)
        
        if target is not None:
            target = self.unnormalize(target)
        
        if target_mask is not None:
            pred, target = pred[target_mask], target[target_mask]
            if self.is_probabilistic:
                mean = self.unnormalize(mean)[target_mask]
                std = std[target_mask]

        if loss_fn is not None:
            if self.is_probabilistic:
                loss = loss_fn(mean, target, std**2, pred)
            else:
                loss = loss_fn(pred, target)

        return pred, target, loss, target_mask
        
class ConvAutoregressiveBaseModel(AutoregressiveBaseModel):

    def predict_steps(self, x, known_features, steps=1, residual=None):
        if self.condition_on_future:
            assert known_features.shape[-1] == (x.shape[-1] + steps)
        if self.predict_week_residual:
            assert residual is not None and residual.shape[-1] == (x.shape[-1] + steps)

        future_conditionals = None
        means, stds = [], []

        for s in range(steps):
            seq_len = x.shape[-1]
            input_conditioning = known_features[..., :seq_len]

            if self.condition_on_future:
                future_conditionals = known_features[..., 1:seq_len+1]

            # add input conditioning
            inputs = torch.cat([x, input_conditioning], dim=CHANNEL_DIM)
            # pass output conditioning
            pred = self(inputs, future_conditionals)
            
            if self.is_probabilistic:
                pred, mean, std = pred
                
            pred = pred[...,-1:]

            if self.predict_week_residual:
                pred = pred + residual[..., seq_len:seq_len+1]
            if self.is_probabilistic:
                mean = mean[...,-1:]
                if self.predict_week_residual:
                    mean = mean + residual[..., seq_len:seq_len+1]
                means.append(mean)
                stds.append(std[...,-1:])
            
            # extend inputs with prediction
            x = torch.cat([x, pred], dim=-1)
            

        if self.is_probabilistic:
            means = torch.cat(means, dim=-1)
            stds = torch.cat(stds, dim=-1)

        return x[..., -steps:].contiguous(), means, stds
    
    def predict_next(self, x, known_features, residual=None):
        seq_len = x.shape[-1]
        input_conditioning = known_features[..., :seq_len]
        future_conditionals = None
        mean, std = None, None

        if self.condition_on_future:
            future_conditionals = known_features[..., 1:seq_len+1]

        # add input conditioning
        inputs = torch.cat([x, input_conditioning], dim=CHANNEL_DIM)
        # pass output conditioning
        pred = self(inputs, future_conditionals)

        if self.is_probabilistic:
            pred, mean, std = pred

        if self.predict_week_residual:
            assert residual is not None and residual.shape[-1] == x.shape[-1] + 1
            pred = pred[..., -1:] + residual[..., -1:]
            if self.is_probabilistic:
                mean = mean[..., -1:] + residual[..., -1:]
                mean = mean.contiguous()
                std = std[..., -1:].contiguous()
        
        return pred.contiguous(), mean, std