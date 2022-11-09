def inference():
    if hasattr(self.model, "run"):
        data = data.to(torch.float32).cpu().numpy()
        # TODO: Should we make this "modelInput configurable", feels complicated
        results = self.model.run(None, {"modelInput": data})[0]
