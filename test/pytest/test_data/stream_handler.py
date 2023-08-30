from ts.torch_handler.base_handler import BaseHandler


class StreamingHandler(BaseHandler):
    def initialize(self, ctx):
        super().initialize(ctx)

        class Stop(object):
            def __init__(self):
                self.counter = 4

            def __call__(self, x):
                self.counter -= 1
                return self.counter == 0

        ctx.stopping_criteria = Stop()

        self.canned_replies = ["hello world ", "hello ", "hello ", "hello "]

    def preprocess(self, data):
        line = data[0]
        text = line.get("data") or line.get("body")
        # Decode text if not a str but bytes or bytearray
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")

        return text

    def inference(self, data):
        return self.model(data)

    def postprocess(self, x):
        return [self.canned_replies.pop()]
