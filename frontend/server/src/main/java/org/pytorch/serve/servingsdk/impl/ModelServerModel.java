package org.pytorch.serve.servingsdk.impl;

import java.util.ArrayList;
import java.util.List;
import org.pytorch.serve.servingsdk.Model;
import org.pytorch.serve.servingsdk.Worker;
import org.pytorch.serve.wlm.ModelManager;

public class ModelServerModel implements Model {
    private final org.pytorch.serve.wlm.Model model;

    public ModelServerModel(org.pytorch.serve.wlm.Model m) {
        model = m;
    }

    @Override
    public String getModelName() {
        return model.getModelName();
    }

    @Override
    public String getModelUrl() {
        return model.getModelUrl();
    }

    @Override
    public String getModelHandler() {
        return model.getModelArchive().getHandler();
    }

    @Override
    public List<Worker> getModelWorkers() {
        ArrayList<Worker> list = new ArrayList<>();
        ModelManager.getInstance()
                .getWorkers(model.getModelVersionName())
                .forEach(r -> list.add(new ModelWorker(r)));
        return list;
    }
}
