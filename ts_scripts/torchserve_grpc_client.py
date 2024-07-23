import argparse
import queue
import threading
from functools import partial

import grpc
import inference_pb2
import inference_pb2_grpc
import management_pb2
import management_pb2_grpc


def get_inference_stub():
    channel = grpc.insecure_channel("localhost:7070")
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub


def get_management_stub():
    channel = grpc.insecure_channel("localhost:7071")
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub


def infer(stub, model_name, model_input, metadata):
    with open(model_input, "rb") as f:
        data = f.read()

    input_data = {"data": data}
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data),
        metadata=metadata,
    )

    try:
        prediction = response.prediction.decode("utf-8")
        print(prediction)
    except grpc.RpcError as e:
        exit(1)


def infer_stream(stub, model_name, model_input, metadata):
    with open(model_input, "rb") as f:
        data = f.read()

    input_data = {"data": data}
    responses = stub.StreamPredictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data),
        metadata=metadata,
    )

    try:
        for resp in responses:
            prediction = resp.prediction.decode("utf-8")
            print(prediction)
    except grpc.RpcError as e:
        exit(1)


def infer_stream2(model_name, sequence_id, input_files, metadata):
    response_queue = queue.Queue()
    process_response_func = partial(
        InferStream2.default_process_response, response_queue
    )

    client = InferStream2SimpleClient()
    try:
        client.start_stream(
            model_name=model_name,
            sequence_id=sequence_id,
            process_response=process_response_func,
            metadata=metadata,
        )
        sequence = input_files.split(",")

        for input_file in sequence:
            client.async_send_infer(input_file.strip())

        for i in range(0, len(sequence)):
            response = response_queue.get()
            print(str(response))

        print("Sequence completed!")

    except grpc.RpcError as e:
        print("infer_stream2 received error", e)
        exit(1)
    finally:
        client.stop_stream()
        client.stop()


def register(stub, model_name, mar_set_str, metadata):
    mar_set = set()
    if mar_set_str:
        mar_set = set(mar_set_str.split(","))
    marfile = f"{model_name}.mar"
    print(f"## Check {marfile} in mar_set :", mar_set)
    if marfile not in mar_set:
        marfile = "https://torchserve.s3.amazonaws.com/mar_files/{}.mar".format(
            model_name
        )

    print(f"## Register marfile: {marfile}\n")
    params = {
        "url": marfile,
        "initial_workers": 1,
        "synchronous": True,
        "model_name": model_name,
    }
    try:
        response = stub.RegisterModel(
            management_pb2.RegisterModelRequest(**params), metadata=metadata
        )
        print(f"Model {model_name} registered successfully")
    except grpc.RpcError as e:
        print(f"Failed to register model {model_name}.")
        print(str(e.details()))
        exit(1)


def unregister(stub, model_name, metadata):
    try:
        response = stub.UnregisterModel(
            management_pb2.UnregisterModelRequest(model_name=model_name),
            metadata=metadata,
        )
        print(f"Model {model_name} unregistered successfully")
    except grpc.RpcError as e:
        print(f"Failed to unregister model {model_name}.")
        print(str(e.details()))
        exit(1)


class InferStream2:
    """
    Create a GRPC bi-directional stream to send and receive inference requests
    and corresponding responses

    :param model_name
    :param sequence_id
    :param process_response: a function with the last parameter response
    """

    def __init__(self, model_name: str, sequence_id: str, process_response):
        self._model_name = model_name
        self._sequence_id = sequence_id
        self._process_response = process_response
        self._request_queue = queue.Queue()
        self._handler = None
        self._alive = True

    def __del__(self):
        self.close()

    def close(self):
        """
        Gracefully close GRPC streams.
        """
        if self._handler is not None:
            self._request_queue.put(None)
            if self._handler.is_alive():
                self._handler.join()
                print("InferStream2 closed")
            self._handler = None

    def init_handler(self, response_iterator):
        if self._handler is not None:
            raise RuntimeError("InferStream2 was already initialized")

        self._handler = threading.Thread(
            target=self._handle_response, args=(response_iterator,)
        )
        self._handler.start()
        print("InferStream2 started")

    def enqueue_request(self, model_input, metadata):
        with open(model_input, "rb") as f:
            data = f.read()

        input_data = {"data": data}
        request = inference_pb2.PredictionsRequest(
            model_name=self._model_name,
            sequence_id=self._sequence_id,
            input=input_data,
            metadata=metadata,
        )
        if self._alive:
            self._request_queue.put(request)
        else:
            raise RuntimeError("The stream is not active.")

    def get_request(self):
        return self._request_queue.get()

    def _handle_response(self, responses):
        try:
            for response in responses:
                self._process_response(response=response)
        except grpc.RpcError as e:
            # The stream is not closed at here.
            self._alive = responses.is_active()
            print("_handle_response exception:", e)
            exit(1)

    @staticmethod
    def default_process_response(
        response_queue: queue.Queue, response: inference_pb2.PredictionResponse
    ):
        if response is not None:
            response_queue.put(response)
        else:
            pass


class RequestIterator:
    """
    An iterator to get a PredictionRequest.

    :param _stream: InferStream2
    """

    def __init__(self, stream: InferStream2):
        self._stream = stream

    def __iter__(self):
        return self

    def __next__(self):
        request = self._stream.get_request()
        if request is None:
            raise StopIteration

        return request


class InferStream2SimpleClient:
    def __init__(self):
        self._stream = None
        self._channel = grpc.insecure_channel("localhost:7070")
        self._stub = inference_pb2_grpc.InferenceAPIsServiceStub(self._channel)

    def start_stream(
        self, model_name: str, sequence_id: str, process_response, metadata
    ):
        if self._stream is not None:
            raise RuntimeError(
                "Cannot start InferStream2SimpleClient since "
                "InferStream2 was already started"
            )

        self._stream = InferStream2(
            model_name=model_name,
            sequence_id=sequence_id,
            process_response=process_response,
        )
        try:
            response_iterator = self._stub.StreamPredictions2(
                RequestIterator(self._stream), metadata=metadata
            )
            self._stream.init_handler(response_iterator)
        except grpc.RpcError as e:
            print("start_stream received error:", e)

    def stop_stream(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def async_send_infer(self, request: str):
        if self._stream is None:
            raise RuntimeError("InferStream2 was already closed")

        self._stream.enqueue_request(request)

    def stop(self):
        self._channel.close()


if __name__ == "__main__":
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "model_name",
        type=str,
        default=None,
        help="Name of the model used.",
    )
    parent_parser.add_argument(
        "--auth-token",
        dest="auth_token",
        type=str,
        default=None,
        required=False,
        help="Authorization token",
    )

    parser = argparse.ArgumentParser(
        description="TorchServe gRPC client",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(help="Action", dest="action")

    infer_action_parser = subparsers.add_parser(
        "infer", parents=[parent_parser], add_help=False
    )
    infer_stream_action_parser = subparsers.add_parser(
        "infer_stream", parents=[parent_parser], add_help=False
    )
    infer_stream2_action_parser = subparsers.add_parser(
        "infer_stream2", parents=[parent_parser], add_help=False
    )
    register_action_parser = subparsers.add_parser(
        "register", parents=[parent_parser], add_help=False
    )
    unregister_action_parser = subparsers.add_parser(
        "unregister", parents=[parent_parser], add_help=False
    )

    infer_action_parser.add_argument(
        "model_input", type=str, default=None, help="Input for model for inference."
    )

    infer_stream_action_parser.add_argument(
        "model_input",
        type=str,
        default=None,
        help="Input for model for stream inference.",
    )

    infer_stream2_action_parser.add_argument(
        "sequence_id",
        type=str,
        default=None,
        help="Input for sequence id for stream inference.",
    )

    infer_stream2_action_parser.add_argument(
        "input_files",
        type=str,
        default=None,
        help="Comma separated list of input files",
    )

    register_action_parser.add_argument(
        "mar_set",
        type=str,
        default=None,
        nargs="?",
        help="Comma separated list of mar models to be loaded using [model_name=]model_location format.",
    )

    args = parser.parse_args()
    if args.auth_token:
        metadata = (
            ("protocol", "gRPC"),
            ("session_id", "12345"),
            ("authorization", f"Bearer {args.auth_token}"),
        )
    else:
        metadata = (("protocol", "gRPC"), ("session_id", "12345"))

    if args.action == "infer":
        infer(get_inference_stub(), args.model_name, args.model_input, metadata)
    elif args.action == "infer_stream":
        infer_stream(get_inference_stub(), args.model_name, args.model_input, metadata)
    elif args.action == "infer_stream2":
        infer_stream2(args.model_name, args.sequence_id, args.input_files, metadata)
    elif args.action == "register":
        register(get_management_stub(), args.model_name, args.mar_set, metadata)
    elif args.action == "unregister":
        unregister(get_management_stub(), args.model_name, metadata)
