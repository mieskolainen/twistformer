# Flow generation and density inference
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import torch
from twistformer.api.contracts import Prediction, Sample
from twistformer.api.predictor import Predictor
from twistformer.randomness import torch_generator


class FlippedConditions:
    def __init__(self, dataset, index):
        """
        Define a binary condition intervention in physical units

        Args:
            dataset:  Raw event dataset
            index:  Binary parameter index
        """
        self.dataset, self.index = dataset, index

    def __len__(self):
        """
        Return the event count
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Flip a binary parameter without changing the source event

        Args:
            index:  Event index
        Returns:
            output: Independent event record
        """
        sample = self.dataset[index]
        parameters = sample.parameters.clone()
        if parameters[self.index] not in (0, 1):
            raise ValueError("Condition flipping requires a binary physical parameter")
        parameters[self.index] = 1 - parameters[self.index]
        return replace(sample, parameters=parameters)


class FlowPredictor(Predictor):
    def condition(self, samples, parameters=None):
        """
        Apply physical hypothesis overrides and optional class flipping

        Args:
            samples:  Raw input events
            parameters:  Optional named physical parameters
        Returns:
            output: Conditioned events
        """
        samples = super().condition(samples, parameters)
        config = self.experiment.config.flow
        if config.mode == "conditional-flip":
            names = self.experiment.driver.input_spec.parameters
            if not config.flip_parameter or config.flip_parameter not in names:
                raise ValueError("conditional-flip requires a declared flow.flip_parameter")
            samples = FlippedConditions(samples, names.index(config.flip_parameter))
        return samples

    @torch.no_grad()
    def forward(self, batch):
        """
        Collect flow losses, density and generated event replicas

        Args:
            batch:  Preprocessed conditions and optional targets
        Returns:
            prediction: Flow results in training coordinates
        """
        self.model.eval()
        result = self.model(batch) if batch.targets is not None and (batch.sample_weights >= 0).all() else Prediction()
        log_prob = result.log_prob
        if log_prob is None and self.model.has_density and batch.targets is not None:
            log_prob = self.model.log_prob(batch)
        if log_prob is not None and batch.flow_log_det is not None:
            log_prob = log_prob + batch.flow_log_det
        samples = self._sample_batch(batch, self.experiment.config.flow.evaluation_samples)
        return replace(result, log_prob=log_prob, samples=samples,
                       mean=samples.mean(1), std=samples.std(1, unbiased=False))

    def _sample_batch(self, batch, count, *, temperature=None, generator=None):
        """
        Apply generation constraints in physical coordinates

        Args:
            batch:  Preprocessed event batch
            count:  Replicas per event
            temperature:  Optional base temperature override
            generator:  Optional random generator
        Returns:
            output: Accepted samples in training coordinates
        """
        config = self.experiment.config.flow
        output = self.model.sample(batch, count=count, temperature=temperature, generator=generator)
        if not config.lower:
            return output
        lower = output.new_tensor(config.lower).reshape(self.model.shape)
        upper = output.new_tensor(config.upper).reshape(self.model.shape)
        for attempt in range(config.max_attempts + 1):
            physical = self.preprocessor.inverse_target(output)
            invalid = ((physical < lower) | (physical > upper) | ~torch.isfinite(physical)).flatten(2).any(-1)
            if not invalid.any():
                return output
            if attempt < config.max_attempts:
                replacement = self.model.sample(batch, count=count, temperature=temperature, generator=generator)
                output = torch.where(invalid.reshape(*invalid.shape, *([1] * len(self.model.shape))), replacement, output)
        raise RuntimeError(f"Physical sampling constraints failed for {int(invalid.sum())} replicas")

    @torch.no_grad()
    def sample(self, samples=None, count=1, *, events=None, batch_size=128, seed=None, temperature=None, parameters=None):
        """
        Generate physical events with reproducible base sampling

        Args:
            samples:  Optional raw conditions and source events
            count:  Replicas per event
            events:  Event count for unconditional generation
            batch_size:  Events per inference batch
            seed:  Base seed override, None uses the checkpoint training seed
            temperature:  Optional base coordinate scale
            parameters:  Optional named physical hypotheses
        Returns:
            output: CPU tensor with batch and replica axes
        """
        if count < 1:
            raise ValueError("Replica count must be positive")
        if samples is None:
            if self.model.conditional or "morph" in self.model.config.mode or events is None or events < 1:
                raise ValueError("Supply input events for conditional generation or a positive unconditional event count")
            width = len(self.experiment.driver.input_spec.features)
            samples = [Sample(torch.zeros(1, width), id=f"generated:{index}") for index in range(events)]
        if seed is None:
            seed = self.experiment.config.training.seed
        generator = torch_generator(seed, device=self.device)
        output = []
        for batch in self.loader(self.condition(samples, parameters), batch_size):
            batch = batch.to(self.device)
            values = self._sample_batch(batch, count, temperature=temperature, generator=generator)
            output.append(self.preprocessor.inverse_target(values).cpu())
        if not output:
            raise ValueError("No events to generate")
        return torch.cat(output)

    @torch.no_grad()
    def log_prob(self, samples, *, batch_size=128, parameters=None):
        """
        Evaluate physical-coordinate densities

        Args:
            samples:  Raw target events and conditions
            batch_size:  Events per density batch
            parameters:  Optional physical hypotheses
        Returns:
            output: CPU log density per event
        """
        output = []
        for batch in self.loader(self.condition(samples, parameters), batch_size):
            batch = batch.to(self.device)
            density = self.model.log_prob(batch)
            if batch.flow_log_det is not None:
                density = density + batch.flow_log_det
            output.append(self.preprocessor.inverse_prediction(Prediction(log_prob=density)).log_prob.cpu())
        if not output:
            raise ValueError("No events for density evaluation")
        return torch.cat(output)

    @torch.no_grad()
    def transport(self, samples, *, batch_size=128, reverse=False, trajectory=False, parameters=None):
        """
        Transport empirical sources or invert observations to base coordinates

        Args:
            samples:  Events with source coordinates, or targets for reverse transport
            batch_size:  Event batch size
            reverse:  Map target coordinates to source
            trajectory:  Return all integration or layer endpoints
            parameters:  Optional physical hypotheses
        Returns:
            output: Physical morphing coordinates or normalized base coordinates
        """
        outputs = []
        morph = "morph" in self.model.config.mode
        if not morph and not reverse:
            raise ValueError("Forward generation uses sample(). Supply base coordinates to model.flow.transport()")
        if not morph and trajectory:
            raise ValueError("Generative reverse trajectories use model.flow.transport() in training coordinates")
        for batch in self.loader(self.condition(samples, parameters), batch_size):
            batch = batch.to(self.device)
            value = batch.targets if reverse else batch.source
            if value is None:
                raise ValueError("Transport needs targets for reverse maps or source coordinates for forward maps")
            if self.model.codec is not None:
                value = self.model.codec.encode(value)[0]
            output = self.model.flow.transport(value, self.model.condition(batch), reverse=reverse, trajectory=trajectory)
            if self.model.codec is not None and morph:
                output = self.model.codec.decode(output)
            outputs.append((self.preprocessor.inverse_target(output) if morph else output).cpu())
        if not outputs:
            raise ValueError("No events to transport")
        return torch.cat(outputs, dim=1 if trajectory else 0)
