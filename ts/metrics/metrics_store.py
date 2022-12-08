

"""
Metrics collection module
"""
from builtins import str

from ts.metrics.dimension import Dimension
from ts.metrics.metric import Metric


class MetricsStore(object):
    """
    DEPRECATED
    Class for creating, modifying different metrics. And keep them in a dictionary
    """

    def __init__(self, request_ids, model_name):
        """
        Initialize metrics map,model name and request map
        """
        self.store = list()
        self.request_ids = request_ids
        self.model_name = model_name
        self.cache = {}

    def _add_or_update(self, name, value, req_id, unit, metrics_method=None, dimensions=None):
        """
        Add a metric key value pair

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        req_id: str
            request id
        unit: str
            unit of metric
        value: int, float , str
            value of metric
        metrics_method: str, optional
            indicates type of metric operation if it is defined
        """
        # IF req_id is none error Metric
        if dimensions is None:
            dimensions = list()
        elif not isinstance(dimensions, list):
            raise ValueError("Please provide a list of dimensions")
        if req_id is None:
            dimensions.append(Dimension("Level", "Error"))
        else:
            dimensions.append(Dimension("ModelName", self.model_name))
            dimensions.append(Dimension("Level", "Model"))

        # Cache the metric with an unique key for update
        dim_str = [name, unit, str(req_id)] + [str(d) for d in dimensions]
        dim_str = '-'.join(dim_str)
        if dim_str not in self.cache:
            metric = Metric(name, value, unit, dimensions, req_id, metrics_method)
            self.store.append(metric)
            self.cache[dim_str] = metric
        else:
            self.cache[dim_str].update(value)

    def _get_req(self, idx):
        """
        Provide the request id dimension

        Parameters
        ----------

        idx : int
            request_id index in batch
        """
        # check if request id for the metric is given, if so use it else have a list of all.
        req_id = self.request_ids
        if isinstance(req_id, dict):
            req_id = ','.join(self.request_ids.values())
        if idx is not None and self.request_ids is not None and idx in self.request_ids:
            req_id = self.request_ids[idx]
        return req_id

    def add_counter(self, name, value, idx=None, dimensions=None):
        """
        Add a counter metric or increment an existing counter metric

        Parameters
        ----------
        name : str
            metric name
        value: int
            value of metric
        idx: int
            request_id index in batch
        dimensions: list
            list of dimensions for the metric
        """
        unit = 'count'
        req_id = self._get_req(idx)
        self._add_or_update(name, value, req_id, unit, 'counter', dimensions)

    def add_time(self, name, value, idx=None, unit='ms', dimensions=None):
        """
        Add a time based metric like latency, default unit is 'ms'

        Parameters
        ----------
        name : str
            metric name
        value: int
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric,  default here is ms, s is also accepted
        dimensions: list
            list of dimensions for the metric
        """
        if unit not in ['ms', 's']:
            raise ValueError("the unit for a timed metric should be one of ['ms', 's']")
        req_id = self._get_req(idx)
        self._add_or_update(name, value, req_id, unit, dimensions)

    def add_size(self, name, value, idx=None, unit='MB', dimensions=None):
        """
        Add a size based metric

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric, default here is 'MB', 'kB', 'GB' also supported
        dimensions: list
            list of dimensions for the metric
        """
        if unit not in ['MB', 'kB', 'GB', 'B']:
            raise ValueError("The unit for size based metric is one of ['MB','kB', 'GB', 'B']")
        req_id = self._get_req(idx)
        self._add_or_update(name, value, req_id, unit, dimensions)

    def add_percent(self, name, value, idx=None, dimensions=None):
        """
        Add a percentage based metric

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        dimensions: list
            list of dimensions for the metric
        """
        unit = 'percent'
        req_id = self._get_req(idx)
        self._add_or_update(name, value, req_id, unit, dimensions)

    def add_error(self, name, value, dimensions=None):
        """
        Add a Error Metric
        Parameters
        ----------
        name : str
            metric name
        value: str
            value of metric, in this case a str
        dimensions: list
            list of dimensions for the metric
        """
        unit = ''

        # noinspection PyTypeChecker
        self._add_or_update(name, value, None, unit, dimensions)

    def add_metric(self, name, value, unit, idx=None, dimensions=None):
        """
        Add a metric which is generic with custom metrics

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric
        dimensions: list
            list of dimensions for the metric
        """
        req_id = self._get_req(idx)
        self._add_or_update(name, value, req_id, unit, dimensions)
