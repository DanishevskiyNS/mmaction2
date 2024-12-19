from typing import Sequence, List
import copy

from mmengine.evaluator import BaseMetric
from mmaction.registry import METRICS

import numpy as np
from sklearn.metrics import fbeta_score

from typing import Any, Callable, List, Optional, Sequence, Union, Tuple, Dict

@METRICS.register_module()  # register the Accuracy class to the METRICS registry
class FBetaScore(BaseMetric):
    """Accuracy evaluation metric."""
    

    def __init__(self,
                 beta: Optional[float] = 1.0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = 'f_beta_score') -> None:
        default_prefix: Optional[str] = f'f{beta}'

        super().__init__(collect_device=collect_device, prefix=prefix)
        self.beta = beta

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_label']
            label = data_sample['gt_label']

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                # single-label
                result['gt'] = label.item()
            else:
                # multi-label
                result['gt'] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # aggregate the classification prediction results and category labels for all samples
        preds = np.array([res['pred'] for res in results])
        gts = np.array([res['gt'] for res in results])


        fbeta_micro_average = fbeta_score(gts, preds, beta=self.beta, average='micro', zero_division=np.nan)
        fbeta_macro_average = fbeta_score(gts, preds, beta=self.beta, average='macro', zero_division=np.nan)

        # return evaluation metric results
        return {f'f{self.beta}_score_micro': fbeta_micro_average, f'{self.beta}_score_macro': fbeta_macro_average}