"""
Module add support for POST, PUT, OPTIONS and DELETE methods to Apache Benchmark
"""
import mimetypes
import os

from math import ceil
from distutils.version import LooseVersion

from bzt import TaurusConfigError
from bzt.modules.aggregator import ConsolidatingAggregator
from bzt.six import iteritems
from bzt.utils import dehumanize_time
from bzt.modules.ab import ApacheBenchmarkExecutor, TSVDataReader


class ApacheBenchmarkExecutor(ApacheBenchmarkExecutor):
    """
    Apache Benchmark executor module
    """

    def prepare(self):
        super(ApacheBenchmarkExecutor, self).prepare()
        self.scenario = self.get_scenario()
        self.install_required_tools()

        self._tsv_file = self.engine.create_artifact("ab", ".tsv")

        self.stdout = open(self.engine.create_artifact("ab", ".out"), 'w')
        self.stderr = open(self.engine.create_artifact("ab", ".err"), 'w')

        self.reader = TSVDataReader(self._tsv_file, self.log)
        if isinstance(self.engine.aggregator, ConsolidatingAggregator):
            self.engine.aggregator.add_underling(self.reader)

    def startup(self):
        args = [self.tool.tool_path]
        load = self.get_load()
        load_iterations = load.iterations or 1
        load_concurrency = load.concurrency or 1

        if load.hold:
            hold = int(ceil(dehumanize_time(load.hold)))
            args += ['-t', str(hold)]
        else:
            args += ['-n', str(load_iterations * load_concurrency)]  # ab waits for total number of iterations

        timeout = self.get_scenario().get("timeout", None)
        if timeout:
            args += ['-s', str(ceil(dehumanize_time(timeout)))]

        args += ['-c', str(load_concurrency)]
        args += ['-d']  # do not print 'Processed *00 requests' every 100 requests or so
        args += ['-r']  # do not crash on socket level errors

        if self.tool.version and LooseVersion(self.tool.version) >= LooseVersion("2.4.7"):
            args += ['-l']  # accept variable-len responses

        args += ['-g', str(self._tsv_file)]  # dump stats to TSV file

        # add global scenario headers
        for key, val in iteritems(self.scenario.get_headers()):
            args += ['-H', "%s: %s" % (key, val)]

        requests = self.scenario.get_requests()
        if not requests:
            raise TaurusConfigError("You must specify at least one request for ab")
        if len(requests) > 1:
            self.log.warning("ab doesn't support multiple requests. Only first one will be used.")
        request = self.__first_http_request()
        if request is None:
            raise TaurusConfigError("ab supports only HTTP requests, while scenario doesn't have any")

        # add request-specific headers
        for key, val in iteritems(request.headers):
            args += ['-H', "%s: %s" % (key, val)]

        # if request.method != 'GET':
        #     raise TaurusConfigError("ab supports only GET requests, but '%s' is found" % request.method)

        if request.method == 'HEAD':
            args += ['-i']
        elif request.method in ['POST', 'PUT']:
            options = {'POST': '-p', 'PUT': '-u'}
            file_path = request.config['file-path']
            if not file_path:
                file_path = os.devnull
                self.log.warning("No file path specified, dev null will be used instead")
            args += [options[request.method], file_path]
            content_type = request.config['content-type'] or mimetypes.guess_type(file_path)[0]
            if content_type:
                args += ['-T', content_type]
        else: # 'GET', 'OPTIONS', 'DELETE', etc
            args += ['-m', request.method]

        if request.priority_option('keepalive', default=True):
            args += ['-k']

        args += [request.url]

        self.reader.setup(load_concurrency, request.label)

        self.log.info('Executing command : ' + ' '.join(arg for arg in args))
        self.process = self._execute(args)


class TSVDataReader(TSVDataReader):
    def _read(self, last_pass=False):
        lines = self.file.get_lines(size=1024 * 1024, last_pass=last_pass)

        for line in lines:
            if not self.skipped_header:
                self.skipped_header = True
                continue
            log_vals = [val.strip() for val in line.split('\t')]

            _error = None
            # _rstatus = None
            _rstatus = '' #Hack to trick taurus into computing aggreated stats

            _url = self.url_label
            _concur = self.concurrency
            _tstamp = int(log_vals[1])  # timestamp - moment of request sending
            _con_time = float(log_vals[2]) / 1000.0  # connection time
            _etime = float(log_vals[4]) / 1000.0  # elapsed time
            _latency = float(log_vals[5]) / 1000.0  # latency (aka waittime)
            _bytes = None

            yield _tstamp, _url, _concur, _etime, _con_time, _latency, _rstatus, _error, '', _bytes
