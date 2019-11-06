import logging
import os
import re
from contextlib import ExitStack
from sockeye import arguments
from sockeye import constants as const
from sockeye import inference
from sockeye.lexicon import TopKLexicon
from sockeye.output_handler import get_output_handler
from sockeye.utils import check_condition, log_basic_info, determine_context

from .model_handler import ModelHandler
from .preprocessor import ChineseCharPreprocessor, Detokenizer


def decode_bytes(data):
    """
    Decodes a bytes array from a file upload

    :param data: a UTF-8 encoded byte array
    :return: a cleaned string
    """
    pattern = re.compile('\r', re.UNICODE)
    res = data.decode('utf-8', 'ignore')
    res = pattern.sub('', res).strip()
    return res


def get_text(req):
    """
    Returns the text string, if any, in the request

    :param req: a JSON request
    :return: a text string
    """
    for field in ['body']:
        if field in req:
            data = req[field]
            if isinstance(data, str):
                return data
            elif isinstance(data, dict) and 'text' in data:
                return data['text']
    return None


def get_file_data(req):
    """
    Returns the file data, if any, in the request

    :param req: a JSON request
    :return: a byte array
    """
    for field in ['body', 'file', 'data']:
        if field in req:
            data = req[field]
            if isinstance(data, bytearray):
                return data
    return None


def read_sockeye_args(params_path):
    """
    Reads command line arguments stored in a file

    :param params_path: path to the parameters file
    :return: a list of command line arguments
    """
    with open(params_path) as f:
        content = f.readlines()

    res = []
    for line in content:
        res += line.split()
    return res


class SockeyeService(ModelHandler):
    """
    Consumes text of arbitrary length and returns its translation.
    """

    def __init__(self):
        super(SockeyeService, self).__init__()
        self.basedir = None
        self.device_ids = []
        self.postprocessor = None
        self.preprocessor = None
        self.sentence_id = 0
        self.translator = None

    def initialize(self, context):
        super(SockeyeService, self).initialize(context)

        self.basedir = context.system_properties.get('model_dir')
        self.preprocessor = ChineseCharPreprocessor(os.path.join(self.basedir, 'bpe.codes.zh-en'),
                                                    os.path.join(self.basedir, 'scripts'),
                                                    os.path.join(self.basedir, 'scripts'))
        self.postprocessor = Detokenizer(os.path.join(self.basedir, 'scripts', 'detokenize.pl'))

        params = arguments.ConfigArgumentParser(description='Translate CLI')
        arguments.add_translate_cli_args(params)

        sockeye_args_path = os.path.join(self.basedir, 'sockeye-args.txt')
        sockeye_args = params.parse_args(read_sockeye_args(sockeye_args_path))
        # override models directory
        sockeye_args.models = [self.basedir]

        if 'gpu_id' in context.system_properties:
            self.device_ids.append(context.system_properties['gpu_id'])
        else:
            logging.warning('No gpu_id found in context')
            self.device_ids.append(0)

        if sockeye_args.checkpoints is not None:
            check_condition(len(sockeye_args.checkpoints) == len(sockeye_args.models),
                            'must provide checkpoints for each model')

        if sockeye_args.skip_topk:
            check_condition(sockeye_args.beam_size == 1, '--skip-topk has no effect if beam size is larger than 1')
            check_condition(len(sockeye_args.models) == 1,
                            '--skip-topk has no effect for decoding with more than 1 model')

        if sockeye_args.nbest_size > 1:
            check_condition(sockeye_args.beam_size >= sockeye_args.nbest_size,
                            'Size of nbest list (--nbest-size) must be smaller or equal to beam size (--beam-size).')
            check_condition(sockeye_args.beam_search_drop == const.BEAM_SEARCH_STOP_ALL,
                            '--nbest-size > 1 requires beam search to only stop after all hypotheses are finished '
                            '(--beam-search-stop all)')
            if sockeye_args.output_type != const.OUTPUT_HANDLER_NBEST:
                logging.warning('For nbest translation, output handler must be "%s", overriding option --output-type.',
                                const.OUTPUT_HANDLER_NBEST)
                sockeye_args.output_type = const.OUTPUT_HANDLER_NBEST

        log_basic_info(sockeye_args)

        output_handler = get_output_handler(sockeye_args.output_type,
                                            sockeye_args.output,
                                            sockeye_args.sure_align_threshold)

        with ExitStack() as exit_stack:
            check_condition(len(self.device_ids) == 1, 'translate only supports single device for now')
            translator_ctx = determine_context(device_ids=self.device_ids,
                                               use_cpu=sockeye_args.use_cpu,
                                               disable_device_locking=sockeye_args.disable_device_locking,
                                               lock_dir=sockeye_args.lock_dir,
                                               exit_stack=exit_stack)[0]
            logging.info('Translate Device: %s', translator_ctx)

            if sockeye_args.override_dtype == const.DTYPE_FP16:
                logging.warning('Experimental feature \'--override-dtype float16\' has been used. '
                                'This feature may be removed or change its behavior in the future. '
                                'DO NOT USE IT IN PRODUCTION')

            models, source_vocabs, target_vocab = inference.load_models(
                context=translator_ctx,
                max_input_len=sockeye_args.max_input_len,
                beam_size=sockeye_args.beam_size,
                batch_size=sockeye_args.batch_size,
                model_folders=sockeye_args.models,
                checkpoints=sockeye_args.checkpoints,
                softmax_temperature=sockeye_args.softmax_temperature,
                max_output_length_num_stds=sockeye_args.max_output_length_num_stds,
                decoder_return_logit_inputs=sockeye_args.restrict_lexicon is not None,
                cache_output_layer_w_b=sockeye_args.restrict_lexicon is not None,
                override_dtype=sockeye_args.override_dtype,
                output_scores=output_handler.reports_score())
            restrict_lexicon = None
            if sockeye_args.restrict_lexicon:
                restrict_lexicon = TopKLexicon(source_vocabs[0], target_vocab)
                restrict_lexicon.load(sockeye_args.restrict_lexicon, k=sockeye_args.restrict_lexicon_topk)
            store_beam = sockeye_args.output_type == const.OUTPUT_HANDLER_BEAM_STORE
            self.translator = inference.Translator(context=translator_ctx,
                                                   ensemble_mode=sockeye_args.ensemble_mode,
                                                   bucket_source_width=sockeye_args.bucket_width,
                                                   length_penalty=inference.LengthPenalty(
                                                       sockeye_args.length_penalty_alpha,
                                                       sockeye_args.length_penalty_beta),
                                                   beam_prune=sockeye_args.beam_prune,
                                                   beam_search_stop=sockeye_args.beam_search_stop,
                                                   nbest_size=sockeye_args.nbest_size,
                                                   models=models,
                                                   source_vocabs=source_vocabs,
                                                   target_vocab=target_vocab,
                                                   restrict_lexicon=restrict_lexicon,
                                                   avoid_list=sockeye_args.avoid_list,
                                                   store_beam=store_beam,
                                                   strip_unknown_words=sockeye_args.strip_unknown_words,
                                                   skip_topk=sockeye_args.skip_topk)

    def preprocess(self, batch):
        """
        Preprocesses a JSON request for translation.

        :param batch: a list of JSON requests of the form { 'text': input_string } or { 'file': file_data }
        :return: a list of input strings to translate
        """
        logging.info('preprocess grabbed: %s' % batch)

        texts = []
        for req in batch:
            data = get_file_data(req)

            if data:
                text = decode_bytes(data)
            else:
                text = get_text(req)

            if text:
                bpe = self.preprocessor.run(text)
                texts.append(bpe)

        return texts

    def inference(self, texts):
        """
        Translates the input data.

        :param texts: a list of strings to translate
        :return: a list of translation objects from Sockeye
        """
        logging.info('inference grabbed: %s' % texts)

        if texts:
            trans_inputs = []
            for t in texts:
                _input = inference.make_input_from_plain_string(self.sentence_id, t)
                trans_inputs.append(_input)
            outputs = self.translator.translate(trans_inputs)

            if len(outputs) != len(trans_inputs):
                logging.warning("Number of translation outputs doesn't match the number of inputs")

            self.sentence_id += len(trans_inputs)
            return outputs
        else:
            self.error = 'Input to inference is empty'
            return []

    def postprocess(self, outputs):
        """
        Converts the translations into a list of JSON responses.

        :param outputs: a list of translation objects from Sockeye
        :return: a list of translations of the form: { 'translation': output_string }
        """
        logging.info('postprocess grabbed: %s' % outputs)

        res = []
        for t in outputs:
            output = self.postprocessor.run(t)
            res.append({'translation': output})
        return res


_service = SockeyeService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
